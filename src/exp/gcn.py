# the physics informed gcn, and its specs
#
# two paths over the same graph:
#   learned  - message passing over every edge, the 93% protein-protein included,
#              which is where the pocket's context comes from
#   physics  - sum(w/r + w/r^6) over the ligand's outward edges only, since the
#              intra-protein and intra-ligand terms cancel in dG_bind
#
# the physics sum reads the raw inv_r / inv_r6 columns by name. never point it at
# log_inv_r6: a log is not an affine rescale, so it would quietly turn
# Lennard-Jones into a log potential.
#
#   gcn_plain          no branch at all
#   gcn_contact_sum    sum(w(class)) - the interface sum, no distance law
#   gcn_physics_blind  sum(w(class)/r + w(class)/r^6) - lennard-jones in earnest
#   gcn_physics        w may read this edge's own r, and w(r)/r is any g(r) it
#                      likes, so the 1/r form constrains nothing
# contact_sum is blind for a reason: with w free to read r, w*1 and w*(1/r) span
# the same functions, so an unblinded contact_sum would just be gcn_physics.

import copy
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import scatter

from de import features
from de.config import N_ELEMENTS, N_NODE_TYPES, N_RESIDUE_TYPES
from de.graph import INTERFACE_EDGES, N_EDGE_TYPES
from exp.spec import ModelSpec

MAX_EPOCHS = 200
# lower patience for memory + time reasons during experiemntation
PATIENCE = 10
GRAD_CLIP = 5.0

DEFAULTS = {
    "hidden": 64,
    "n_layers": 3,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "physics": True,
    # the ablations at the top of this file; both off is gcn_physics
    "physics_blind": False,
    "physics_constant": False,
}


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class PhysicsGCN(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        edge_feature_names,
        hidden=64,
        n_layers=3,
        dropout=0.1,
        physics=True,
        physics_blind=False,
        physics_constant=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.physics = physics
        # blind: the pair mlp is handed the edge class but not this edge's
        # distances. constant: the sum drops the 1/r, 1/r^6 terms altogether
        self.physics_blind = physics_blind
        self.physics_constant = physics_constant

        self.element = nn.Embedding(N_ELEMENTS, hidden)
        self.node_type = nn.Embedding(N_NODE_TYPES, hidden)
        # helps differentiate same atom in different residue
        self.residue_type = nn.Embedding(N_RESIDUE_TYPES, hidden)
        self.node_proj = nn.Linear(node_feat_dim, hidden)
        self.edge_type = nn.Embedding(N_EDGE_TYPES, hidden)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden)

        # LayerNorm plus a residual per conv. GINEConv sums over neighbours and
        # the 4 angstrom graph has mean degree 11, so unnormalised activations grow lik11x per layer
        self.convs = nn.ModuleList(
            GINEConv(_mlp(hidden, hidden, hidden)) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.head = _mlp(hidden, hidden, 1)

        if physics:
            # by name, so the order in edge_attr is not load bearing and a
            # missing feature fails here rather than silently
            missing = set(features.PHYSICS_EDGE_FEATURES) - set(edge_feature_names)
            if missing:
                raise ValueError(f"physics branch needs edge features {sorted(missing)}")
            columns = [
                edge_feature_names.index(name)
                for name in features.PHYSICS_EDGE_FEATURES
            ]
            # buffers, so neither is rebuilt from a python list every forward
            self.register_buffer("physics_columns", torch.tensor(columns))
            self.register_buffer("interface_edges", torch.tensor(INTERFACE_EDGES))

            # per-pair coefficients for (1/r, 1/r^6) from both endpoint
            # embeddings plus the edge class, so metal coordination can take a
            # different coefficient from an ordinary contact
            self.pair = _mlp(3 * hidden, hidden, len(columns))
            self.energy_scale = nn.Parameter(torch.zeros(1))

    def _node_embedding(self, data):
        # summing embeddings plus a projection of x is the same linear map as
        # concatenating the one-hots with x through one bias-free Linear, just
        # factorised into row lookups
        return (
            self.element(data.atomic_num)
            + self.node_type(data.node_type)
            + self.residue_type(data.residue_type)
            + self.node_proj(data.x)
        )

    def _energy(self, data, h, pair_embedding, batch, n_graphs):
        """
        sum(w1/r + w2/r^6) over interface edges, per graph.

        `pair_embedding` is what the coefficients may read of each edge: the full
        embedding, or the class alone when blind. Halved because add_edges stores
        both directions: correct for message passing, a factor of two for an
        energy. scatter needs dim_size or a graph with no interface edge drops
        out and everything after it misaligns with y.
        """
        # one index tensor, so the three reads below do not each re-scan a mask.
        # empty needs no special case - scatter with dim_size returns zeros
        edges = torch.isin(data.edge_type, self.interface_edges).nonzero().squeeze(-1)

        src, dst = data.edge_index[:, edges]
        coefficients = self.pair(
            torch.cat([h[src], h[dst], pair_embedding[edges]], dim=1)
        )
        if self.physics_constant:
            # contact counting: the coefficient is the whole contribution, so
            # edge_attr is not read at all down this path
            terms = coefficients.new_ones(coefficients.shape)
        else:
            terms = data.edge_attr[edges][:, self.physics_columns]
        energy = 0.5 * (coefficients * terms).sum(dim=1)
        return scatter(energy, batch[src], dim=0, dim_size=n_graphs, reduce="sum")

    def forward(self, data):
        # a bare Data has no .batch - kept so model(graph) works for a one-off
        # check outside a DataLoader, which is how the branch gets exercised
        batch = data.batch
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        n_graphs = int(batch.max()) + 1

        h = self._node_embedding(data)
        # kept apart so the physics branch can be handed the class alone; message
        # passing always gets the full embedding, distances included
        edge_type_embedding = self.edge_type(data.edge_type)
        edge_embedding = edge_type_embedding + self.edge_proj(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            message = F.relu(norm(conv(h, data.edge_index, edge_embedding)))
            h = h + F.dropout(message, p=self.dropout, training=self.training)

        out = self.head(global_mean_pool(h, batch)).squeeze(-1)
        if self.physics:
            # energy_scale starts at 0 so the physics term fades in rather than
            # swamping the learned branch while its coefficients are random
            out = out + self.energy_scale * self._energy(
                data,
                h,
                edge_type_embedding if self.physics_blind else edge_embedding,
                batch,
                n_graphs,
            )
        return out


def _build(params, dims, seed=0, edge_feature_names=None):
    """edge_feature_names comes from the spec via evaluate(), never defaulted"""
    torch.manual_seed(seed)
    return PhysicsGCN(
        node_feat_dim=dims["node_feat_dim"],
        edge_feat_dim=dims["edge_feat_dim"],
        edge_feature_names=edge_feature_names,
        hidden=params["hidden"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        physics=params["physics"],
        physics_blind=params.get("physics_blind", False),
        physics_constant=params.get("physics_constant", False),
    )


def _epoch(model, loader, target, optimizer=None):
    """one pass; optimizer=None scores instead of training. returns mean loss"""
    model.train(optimizer is not None)
    total, n = 0.0, 0
    for batch in loader:
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(batch)
            loss = F.mse_loss(prediction, target.encode(batch.y))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            # dense graphs and summed aggregation make the occasional huge
            # gradient; clipping stops one batch destroying the run
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def _fit(model, train, val, params, target):
    """
    Adam with early stopping on val, restoring the best weights.

    `target` was fitted on train, so val contributes nothing but when to stop.
    The per-epoch curve lands on model.history, and best_epoch / best_val_loss
    beside it: losses in encoded units, so comparable only within a run.
    """
    max_epochs = params.get("max_epochs", MAX_EPOCHS)
    patience = params.get("patience", PATIENCE)
    train_loader = DataLoader(train, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=params["batch_size"]) if val else None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    progress = params.get("progress", False)
    model.history = history = []
    model.best_epoch, model.best_val_loss = None, float("inf")
    best_state, waited = None, 0
    for epoch_index in range(max_epochs):
        started = time.perf_counter()
        train_loss = _epoch(model, train_loader, target, optimizer)
        val_loss = _epoch(model, val_loader, target) if val_loader else float("nan")
        history.append(
            {
                "epoch": epoch_index,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        if val_loader is None:
            continue
        if val_loss < model.best_val_loss:
            model.best_epoch, model.best_val_loss = epoch_index, val_loss
            best_state, waited = copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
        if progress:
            print(
                f"    epoch {epoch_index:>3}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  best {model.best_val_loss:.4f}"
                f"@{model.best_epoch}  waited {waited}  "
                f"[{history[-1]['seconds']:.1f}s]",
                flush=True,
            )
        if waited >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, graphs, target):
    model.eval()
    true, predicted = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=64):
            predicted.append(target.decode(model(batch)))
            true.append(batch.y)
    return torch.cat(true).numpy(), torch.cat(predicted).numpy()


def _spec(name, **overrides):
    """a spec with DEFAULTS overridden"""
    edge_features = overrides.pop("edge_features", features.DEFAULT_EDGE_FEATURES)
    return ModelSpec(
        name=name,
        build=_build,
        fit=_fit,
        predict=_predict,
        params={**DEFAULTS, **overrides},
        edge_features=edge_features,
    )


# the variants worth comparing first: does the physics branch earn its place?
GCN_PHYSICS = _spec("gcn_physics", physics=True)
GCN_PLAIN = _spec("gcn_plain", physics=False)

# and which part of it earns the place - the ladder at the top of this file
GCN_PHYSICS_BLIND = _spec("gcn_physics_blind", physics=True, physics_blind=True)
GCN_CONTACT_SUM = _spec(
    "gcn_contact_sum", physics=True, physics_blind=True, physics_constant=True
)

# the capacity control for the plain
GCN_PLAIN_WIDE = _spec("gcn_plain_wide", physics=False, hidden=75)

MODELS = {
    spec.name: spec
    for spec in (
        GCN_PHYSICS,
        GCN_PLAIN,
        GCN_PHYSICS_BLIND,
        GCN_CONTACT_SUM,
        GCN_PLAIN_WIDE,
    )
}

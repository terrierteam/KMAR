import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from tqdm import tqdm

# ------------------------- GraphAttentionLayer -------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W)
        We = torch.matmul(entity_embs, self.W)

        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs.unsqueeze(0).expand(attention.size(0), -1, -1)).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh_expanded = Wh.unsqueeze(1)
        We_expanded = We.unsqueeze(0).expand(Wh.size(0), We.size(0), -1)
        return torch.cat((Wh_expanded.expand(-1, We.size(0), -1), We_expanded), dim=-1)

# ------------------------- Utility Functions -------------------------
def load_processed_data_with_mapping(id_path, text_path):
    id_data = pd.read_csv(id_path, sep="\t", header=None, names=["item", "relation", "entity"])
    text_data = pd.read_csv(text_path, sep="\t", header=None, names=["item", "relation", "entity"])

    if len(id_data) != len(text_data):
        raise ValueError("no")

    unique_items = id_data["item"].unique()
    unique_entities = id_data["entity"].unique()

    item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    entity_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_entities)}

    id_data["item"] = id_data["item"].map(item_mapping)
    id_data["entity"] = id_data["entity"].map(entity_mapping)

    return id_data, text_data, item_mapping, entity_mapping

def initialize_embeddings(num_items, num_relations, num_entities, embedding_dim):
    print(f"Initializing embeddings: num_items={num_items}, num_entities={num_entities}")
    item_embeddings = nn.Embedding(num_items, embedding_dim)
    relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    entity_embeddings = nn.Embedding(num_entities, embedding_dim)

    nn.init.xavier_uniform_(item_embeddings.weight)
    nn.init.xavier_uniform_(relation_embeddings.weight)
    nn.init.xavier_uniform_(entity_embeddings.weight)

    return item_embeddings, relation_embeddings, entity_embeddings

# ------------------------- Loss Function -------------------------
def contrastive_loss(item_embeddings, entity_embeddings, adj, margin=1.0):
    pos_pairs = adj.nonzero()
    neg_pairs = torch.randint(0, adj.shape[0], pos_pairs.shape, device=item_embeddings.device)

    pos_dist = torch.norm(item_embeddings[pos_pairs[:, 0]] - entity_embeddings[pos_pairs[:, 1]], dim=1)
    neg_dist = torch.norm(item_embeddings[neg_pairs[:, 0]] - entity_embeddings[neg_pairs[:, 1]], dim=1)

    loss = F.relu(margin + pos_dist - neg_dist).mean()
    return loss

# ------------------------- EarlyStopping  -------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------- Pretraining  -------------------------
def pretrain_model(item_embeddings, entity_embeddings, adj, gat_layer, epochs=500, lr=0.001, patience=10):
    optimizer = torch.optim.Adam(list(gat_layer.parameters()) + [item_embeddings.weight, entity_embeddings.weight], lr=lr)
    adj_dense = adj.to_dense()

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)

    for epoch in range(epochs):
        optimizer.zero_grad()

        gat_output = gat_layer(item_embeddings.weight, entity_embeddings.weight, adj_dense)
        loss = contrastive_loss(item_embeddings.weight, entity_embeddings.weight, adj_dense)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break
            
# ------------------------- Similarity Computation -------------------------
def compute_item_entity_similarity_in_batches(item_embeddings, entity_embeddings, adj, gat_layer, batch_size=5, chunk_size=50):
    similarity_scores = []
    num_items = item_embeddings.size(0)
    num_entities = entity_embeddings.size(0)

    adj_dense = adj.to_dense()

    for start in tqdm(range(0, num_items, batch_size), desc="Processing Batches"):
        end = min(start + batch_size, num_items)
        batch_item_embeddings = item_embeddings[start:end]
        batch_adj = adj_dense[start:end]

        similarity_chunk = []
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            batch_entity_embeddings = entity_embeddings[chunk_start:chunk_end]
            batch_chunk_adj = batch_adj[:, chunk_start:chunk_end]

            batch_scores = gat_layer(batch_item_embeddings, batch_entity_embeddings, batch_chunk_adj)
            similarity_chunk.append(batch_scores)

        similarity_scores.append(torch.cat(similarity_chunk, dim=1))

    similarity_scores = torch.cat(similarity_scores, dim=0) 
    assert similarity_scores.size(0) == num_items, "Mismatch in similarity scores length!"
    return similarity_scores

# ------------------------- Extract Top Triples -------------------------
def get_top_triples(similarity_scores, id_data, item_mapping, entity_mapping):
    top_triples_id = []
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    reverse_entity_mapping = {v: k for k, v in entity_mapping.items()}

    unique_items = id_data['item'].unique()

    for mapped_item_id in tqdm(unique_items, desc="Processing items"):
        item_id = reverse_item_mapping[mapped_item_id]
        item_relations = id_data[id_data['item'] == mapped_item_id]

        valid_entities = item_relations['entity'].tolist()
        relations = item_relations['relation'].tolist()

        entity_scores = similarity_scores[mapped_item_id, :len(valid_entities)]

        top_idx = torch.argmax(entity_scores).item()
        top_entity_id = valid_entities[top_idx]
        top_relation_id = relations[top_idx]

        original_entity_id = reverse_entity_mapping[top_entity_id]
        original_relation_id = top_relation_id

        top_triples_id.append((item_id, original_relation_id, original_entity_id))

    return top_triples_id

# ------------------------- Extract Top two Triples -------------------------
def get_top_two_triples(similarity_scores, id_data, item_mapping, entity_mapping):
    top_triples_id = []
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    reverse_entity_mapping = {v: k for k, v in entity_mapping.items()}

    unique_items = id_data['item'].unique()

    for mapped_item_id in tqdm(unique_items, desc="Processing items"):
        item_id = reverse_item_mapping[mapped_item_id]
        item_relations = id_data[id_data['item'] == mapped_item_id]

        valid_entities = item_relations['entity'].tolist()
        relations = item_relations['relation'].tolist()

        if len(valid_entities) == 0:
            continue

        entity_scores = similarity_scores[mapped_item_id, :len(valid_entities)]

        k = min(2, len(entity_scores)) 
        top_indices = torch.topk(entity_scores, k=k).indices 

        for idx in top_indices:
            top_entity_id = valid_entities[idx]
            top_relation_id = relations[idx]

            original_entity_id = reverse_entity_mapping[top_entity_id]
            original_relation_id = top_relation_id

            top_triples_id.append((item_id, original_relation_id, original_entity_id))

    return top_triples_id

def save_triples_to_files(top_triples_id, id_output_path):
    id_df = pd.DataFrame(top_triples_id, columns=["item", "relation", "entity"])
    id_df.to_csv(id_output_path, sep="\t", index=False, header=False)
    print(f"Saved ID triples to {id_output_path}")

# ------------------------- Main Function -------------------------
def main():
    device = torch.device("cpu")  
    print(f"Running on: {device}")

    id_path = "processed_kg_id.tsv"
    embedding_dim = 16

    id_data, _, item_mapping, entity_mapping = load_processed_data_with_mapping(id_path, id_path)
    num_items = len(item_mapping)
    num_entities = len(entity_mapping)

    print(f"Initializing embeddings: num_items={num_items}, num_entities={num_entities}")
    item_embeddings, _, entity_embeddings = initialize_embeddings(
        num_items, len(id_data["relation"].unique()), num_entities, embedding_dim
    )
    item_embeddings = item_embeddings.to(device)
    entity_embeddings = entity_embeddings.to(device)

    rows, cols, data = [], [], []
    for _, row in id_data.iterrows():
        rows.append(row["item"])
        cols.append(row["entity"])
        data.append(1)
    adj_sparse = coo_matrix((data, (rows, cols)), shape=(num_items, num_entities))
    adj_dense = torch.sparse_coo_tensor(
        torch.tensor([adj_sparse.row, adj_sparse.col], dtype=torch.long),
        torch.tensor(adj_sparse.data, dtype=torch.float),
        torch.Size(adj_sparse.shape)
    ).to(device)

    print(f"Adjacency matrix shape: {adj_dense.shape}")
    gat_layer = GraphAttentionLayer(embedding_dim, embedding_dim, dropout=0.5, alpha=0.2, concat=False).to(device)

    pretrain_model(item_embeddings, entity_embeddings, adj_dense, gat_layer, epochs=500, lr=0.001, patience=10)


    similarity_scores = compute_item_entity_similarity_in_batches(
        item_embeddings.weight, 
        entity_embeddings.weight, 
        adj_dense, 
        gat_layer, 
        batch_size=10,
        chunk_size=50
    )
    print(f"Similarity scores shape: {similarity_scores.shape}")

    top_triples_id = get_top_triples(similarity_scores, id_data, item_mapping, entity_mapping)
    
    id_output_path = "pretrain-output_kg_id.tsv"
    save_triples_to_files(top_triples_id, id_output_path)

if __name__ == "__main__":
    main()

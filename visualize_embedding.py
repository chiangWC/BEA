import torch
from train import BEA, extract_embeddings
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(embeddings, labels, label_names=None, title='Embedding Visualization'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for l in unique_labels:
        idx = labels == l
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label_names[l] if label_names else str(l), alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型参数
    from train import init_args
    args = init_args()
    model = BEA(args, output_dim=4).to(device)
    model.load_state_dict(torch.load('your_model_path.pt', map_location=device))
    model.eval()

    # 加载数据
    conversation_history = torch.load(f'./BEA/tensor/{args.model_name}_conversation_tensor.pt')
    tutor_responses = torch.load(f'./BEA/tensor/{args.model_name}_response_tensor.pt')
    label = torch.load(f'./BEA/tensor/{args.model_name}_label_tensor.pt')
    dataset = TensorDataset(conversation_history, tutor_responses, label)
    dataloader = DataLoader(dataset, batch_size=64)

    # 提取 embedding 和标签
    embeddings, labels = extract_embeddings(model, dataloader, device)

    # 可视化
    plot_embeddings(embeddings, labels, label_names=['Yes', 'To some extent', 'No'])
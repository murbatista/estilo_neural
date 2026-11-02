import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# 1. Configuração de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando no dispositivo: {device}")

# 2. Pré-processamento e Carregamento
imsize = 512 if torch.cuda.is_available() else 256

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 3. Funções de Perda (Loss Functions)
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class MultiStyleLoss(nn.Module):
    def __init__(self, target_features_list):
        super(MultiStyleLoss, self).__init__()
        # Calcula a média das Matrizes de Gram de TODAS as fotos da pasta
        gram_list = [gram_matrix(feat).detach() for feat in target_features_list]
        self.target = torch.stack(gram_list).mean(dim=0)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# 4. Carregamento da VGG19
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 5. Carregar Todas as Imagens da Pasta de Estilo
PASTA_ESTILOS = "minha_base_estilos"
extensoes = ("*.jpg", "*.jpeg", "*.png")
caminhos_estilos = []

for ext in extensoes:
    caminhos_estilos.extend(glob.glob(os.path.join(PASTA_ESTILOS, ext)))

if not caminhos_estilos:
    print(f"⚠️ Nenhuma imagem encontrada na pasta '{PASTA_ESTILOS}'!")
    exit()

print(f"✓ Carregando {len(caminhos_estilos)} fotos de estilo da pasta...")
style_imgs = [image_loader(p) for p in caminhos_estilos]
content_img = image_loader("emkanicepic-borgward-6621359.jpg")

# 6. Montagem do Modelo com Múltiplos Estilos
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_imgs, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Extrai as características de TODAS as imagens da pasta para essa camada
            target_features = [model(s_img).detach() for s_img in style_imgs]
            style_loss = MultiStyleLoss(target_features)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], MultiStyleLoss):
            break

    model = model[:i+1]
    return model, style_losses, content_losses

# 7. Processo de Otimização
input_img = content_img.clone()

print("Construindo o modelo de transferência multi-estilo...")
model, style_losses, content_losses = get_style_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std, style_imgs, content_img
)

optimizer = optim.LBFGS([input_img.requires_grad_()])

num_steps = 300
style_weight = 1000000
content_weight = 1

run = [0]
print("Iniciando a otimização com a base de fotos...")

while run[0] <= num_steps:
    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Passo {run[0]}/{num_steps} | Perda Estilo: {style_score.item():.4f} | Perda Conteúdo: {content_score.item():.4f}")

        return style_score + content_score

    optimizer.step(closure)

with torch.no_grad():
    input_img.clamp_(0, 1)

# 8. Salvando a Imagem Final
unloader = transforms.ToPILImage()
image = input_img.cpu().clone().squeeze(0)
image = unloader(image)
image.save("resultado_combina_pasta.jpg")

print("\n🎉 Imagem final gerada com sucesso e salva como 'resultado_combina_pasta.jpg'!")
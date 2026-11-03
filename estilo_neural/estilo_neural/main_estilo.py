
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import transforms, models


# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DO DISPOSITIVO
# -------------------------------------------------------------

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo utilizado: {dispositivo}")

tamanho_imagem = 512 if torch.cuda.is_available() else 256


# -------------------------------------------------------------
# 2. PREPARAÇÃO DAS IMAGENS
# -------------------------------------------------------------

transformacao = transforms.Compose([
    transforms.Resize((tamanho_imagem, tamanho_imagem)),
    transforms.ToTensor()
])


def carregar_imagem(caminho):

    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")
        imagem = transformacao(imagem).unsqueeze(0)

    return imagem.to(dispositivo, torch.float)


# -------------------------------------------------------------
# 3. CÁLCULO DAS PERDAS DE CONTEÚDO E ESTILO
# -------------------------------------------------------------

class PerdaConteudo(nn.Module):

    def __init__(self, imagem_referencia):
        super().__init__()

        self.referencia = imagem_referencia.detach()

    def forward(self, entrada):

        self.loss = nn.functional.mse_loss(
            entrada, self.referencia
        )

        return entrada


def calcular_gram(matriz):

    lote, canais, altura, largura = matriz.size()

    caracteristicas = matriz.view(
        lote * canais,
        altura * largura
    )

    gram = torch.mm(
        caracteristicas,
        caracteristicas.t()
    )

    return gram / (lote * canais * altura * largura)


class PerdaMultiplosEstilos(nn.Module):

    def __init__(self, caracteristicas_estilo):
        super().__init__()

        matrizes = [
            calcular_gram(caracteristica).detach()
            for caracteristica in caracteristicas_estilo
        ]

        self.referencia = torch.stack(matrizes).mean(dim=0)

    def forward(self, entrada):

        matriz_atual = calcular_gram(entrada)

        self.loss = nn.functional.mse_loss(
            matriz_atual, self.referencia
        )

        return entrada


# -------------------------------------------------------------
# 4. CARREGAMENTO DA REDE NEURAL VGG19
# -------------------------------------------------------------

rede_vgg = models.vgg19(
    weights=models.VGG19_Weights.DEFAULT
).features.to(dispositivo).eval()


media = torch.tensor(
    [0.485, 0.456, 0.406]
).to(dispositivo)

desvio = torch.tensor(
    [0.229, 0.224, 0.225]
).to(dispositivo)


class NormalizarImagem(nn.Module):

    def __init__(self, media, desvio):
        super().__init__()

        self.media = media.view(-1, 1, 1)
        self.desvio = desvio.view(-1, 1, 1)

    def forward(self, imagem):

        return (imagem - self.media) / self.desvio


camadas_conteudo = ["conv_4"]

camadas_estilo = [
    "conv_1",
    "conv_2",
    "conv_3",
    "conv_4",
    "conv_5"
]


# -------------------------------------------------------------
# 5. LOCALIZAÇÃO DAS IMAGENS DE REFERÊNCIA
# -------------------------------------------------------------

diretorio_estilos = "minha_base_estilos"

formatos_imagem = ("*.jpg", "*.jpeg", "*.png")

imagens_encontradas = []

for formato in formatos_imagem:

    imagens_encontradas.extend(
        glob.glob(os.path.join(diretorio_estilos, formato))
    )


if not imagens_encontradas:

    print(
        f"Nenhuma imagem encontrada em '{diretorio_estilos}'!"
    )

    exit()


print(
    f"\nForam encontradas {len(imagens_encontradas)} imagens de estilo."
)

imagens_estilo = [
    carregar_imagem(caminho)
    for caminho in imagens_encontradas
]

imagem_original = carregar_imagem(
    "emkanicepic-borgward-6621359.jpg"
)


# -------------------------------------------------------------
# 6. CONSTRUÇÃO DO MODELO DE TRANSFERÊNCIA DE ESTILO
# -------------------------------------------------------------

def construir_modelo(rede, media, desvio,
                     estilos, conteudo):

    normalizacao = NormalizarImagem(
        media, desvio
    ).to(dispositivo)

    modelo = nn.Sequential(normalizacao)

    perdas_estilo = []
    perdas_conteudo = []

    contador = 0

    for camada in rede.children():

        if isinstance(camada, nn.Conv2d):

            contador += 1
            nome = f"conv_{contador}"

        elif isinstance(camada, nn.ReLU):

            nome = f"relu_{contador}"
            camada = nn.ReLU(inplace=False)

        elif isinstance(camada, nn.MaxPool2d):

            nome = f"pool_{contador}"

        elif isinstance(camada, nn.BatchNorm2d):

            nome = f"bn_{contador}"

        else:
            continue

        modelo.add_module(nome, camada)

        # Extração das características de conteúdo
        if nome in camadas_conteudo:

            referencia = modelo(conteudo).detach()

            perda = PerdaConteudo(referencia)

            modelo.add_module(
                f"perda_conteudo_{contador}", perda
            )

            perdas_conteudo.append(perda)

        # Extração das características de todas as imagens de estilo
        if nome in camadas_estilo:

            caracteristicas = [
                modelo(estilo).detach()
                for estilo in estilos
            ]

            perda = PerdaMultiplosEstilos(caracteristicas)

            modelo.add_module(
                f"perda_estilo_{contador}", perda
            )

            perdas_estilo.append(perda)

    # Remove as camadas posteriores às últimas perdas
    for indice in range(len(modelo) - 1, -1, -1):

        if isinstance(
            modelo[indice],
            (PerdaConteudo, PerdaMultiplosEstilos)
        ):
            break

    modelo = modelo[:indice + 1]

    return modelo, perdas_estilo, perdas_conteudo


# -------------------------------------------------------------
# 7. OTIMIZAÇÃO DA IMAGEM
# -------------------------------------------------------------

imagem_gerada = imagem_original.clone()

print("\nMontando o modelo de transferência de estilo...")

modelo, perdas_estilo, perdas_conteudo = construir_modelo(
    rede_vgg,
    media,
    desvio,
    imagens_estilo,
    imagem_original
)

otimizador = optim.LBFGS([
    imagem_gerada.requires_grad_()
])

total_etapas = 300

peso_estilo = 1000000
peso_conteudo = 1

etapa_atual = [0]

print("\nIniciando o processo de combinação das imagens...")


while etapa_atual[0] <= total_etapas:

    def calcular_perda():

        with torch.no_grad():
            imagem_gerada.clamp_(0, 1)

        otimizador.zero_grad()

        modelo(imagem_gerada)

        valor_estilo = 0
        valor_conteudo = 0

        for perda in perdas_estilo:
            valor_estilo += perda.loss

        for perda in perdas_conteudo:
            valor_conteudo += perda.loss

        valor_estilo *= peso_estilo
        valor_conteudo *= peso_conteudo

        perda_total = valor_estilo + valor_conteudo

        perda_total.backward()

        etapa_atual[0] += 1

        if etapa_atual[0] % 50 == 0:

            print(
                f"Etapa {etapa_atual[0]}/{total_etapas} | "
                f"Estilo: {valor_estilo.item():.4f} | "
                f"Conteúdo: {valor_conteudo.item():.4f}"
            )

        return perda_total

    otimizador.step(calcular_perda)


# -------------------------------------------------------------
# 8. EXPORTAÇÃO DA IMAGEM RESULTANTE
# -------------------------------------------------------------

with torch.no_grad():
    imagem_gerada.clamp_(0, 1)

converter_imagem = transforms.ToPILImage()

resultado = imagem_gerada.cpu().clone().squeeze(0)

resultado = converter_imagem(resultado)

nome_resultado = "resultado_combina_pasta.jpg"

resultado.save(nome_resultado)

print("\nProcesso concluído com sucesso!")
print(f"Imagem salva como: {os.path.abspath(nome_resultado)}")

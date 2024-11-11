# Visão Geral

**Comprehensive Toolbox:** MMDet is a comprehensive toolbox for object detection, instance segmentation, and pose estimation tasks.

**Open Source:** It is an open-source project hosted on GitHub allowing for community contributions and improvements.

**Built on PyTorch:** Developed using the PyTorch deep learning framework, ensuring ease of use and customizability.

**Modular Design:** Features a modular design that enables the quick assembly of different components to experiment with new ideas.

**State-of-the-Art Models:** Provides implementations of state-of-the-art models including Faster R-CNN, Mask R-CNN, YOLOv3, and many others.

**Extensive Model Zoo:** Comes with a large model zoo offering pre-trained models on various datasets such as COCO, PASCAL VOC, and Cityscapes.

# Estrutura da Configuração MMDet

O MMDet utiliza um sistema hierárquico de configuração para flexibilidade:

- **Arquivos Base:** Você possui uma coleção de arquivos de configuração base no diretório `../_base_`. Eles estabelecem a estrutura central e padrões para vários componentes:
  - `custom_imports.py:` Gerencia qualquer módulo ou código personalizado necessário.
  - `default_runtime.py:` Configurações padrão de tempo de execução (por exemplo, registro onde os checkpoints são salvos).
  - `schedule_120e.py:` Define o cronograma de treinamento (como a taxa de aprendizado muda ao longo do tempo).
  - `indoor360.py:` Especifica o dataset Indoor360 a ser usado.
  - `sph_retinanet_r50_fpn.py:` Define a arquitetura principal - um modelo RetinaNet com uma rede backbone ResNet-50 e Feature Pyramid Network (FPN).

**Configuração Principal:** Seu arquivo de configuração principal reúne esses arquivos base usando `_base_ = [...]`. Isso cria um ponto de partida.

**Sobrescrevendo:** A configuração principal pode então personalizar as coisas. Subscreve determinados valores dos arquivos base:
  - `checkpoint_config` e `evaluation:` Modifica a frequência com que os checkpoints são salvos e as avaliações são executadas.
  - `log_config:` Modifica a saída do log.
  - `data:` Ajusta o tamanho do batch e o processamento de dados.
  - `model:` Redefine partes da estrutura do modelo provenientes de `sph_retinanet_r50_fpn.py.`

# 1 - Acionamento do Dev Container


gambiarra pra usar git: git config --global --add safe.directory /home/mstveras/mmdetection_2


**IMPORTANTE:** A “parte esférica” do código foi escrita em uma versão antiga do mmdet (acredito que não seja muito difícil atualizar para a mais recente). Tentei instalar as bibliotecas normalmente pelo pip, mas esse processo se mostrou muito penoso. Optei por utilizar um Dockerfile especificando o que precisávamos. Também utilizei a extensão de dev containers do vscode que, na minha visão, simplifica bastante o processo também - mas essa é totalmente opcional. Por fim, tive que usar uma gambiarra que consiste em baixar as seguintes bibliotecas depois de instanciar o container:  

Após instanciar o dev container falta baixar as bibliotecas com:

```bash
pip install -v -e .
pip install yapf==0.40.1
pip install future tensorboard
```

line profiler:

apt-get update
pip install line_profiler



Quebrei um pouco a cabeça para incluir esses comandos no Dockerfile para que estivessem incluídos assim que instanciasse a imagem, mas nada deu certo. Como a gambiarra deu conta do recado, acabei deixando assim mesmo.

# 2 - Comandos do MMdet

**Rodar treinamento:**

- **RetinaNet:**
  ```bash
  python3 tools/train.py configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py
  ```

- **Faster R-CNN:**
  ```bash
  python3 tools/train.py configs/faster_rcnn/sph_faster_rcnn_r50_fpn.py
  ```

**Rodar teste e gerar relatório com mean avg precisions** (alternativamente pode-se usar flag recall para cálculo do recall):

```bash
python3 tools/test.py configs/retinanet/retinanet_r50_fpn_fp16_1x_coco.py checkpoints/best_bbox_mAP_50_epoch_85.pth  --eval mAP
```

**Rodar teste e gerar arquivo pickle** (Necessário para gerar imagens de predição):

```bash
python3 tools/test.py configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py work_dirs/sph_retinanet_r50_fpn_120e_indoor360/<epoch de interesse> --out <nome arquivo pickle gerado>
```

**Gerar imagens da predição:**

```bash
python3 tools/analysis_tools/analyze_results.py configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py <arquivo-pickle> <dir salvar-imagens>
```

# SPHDET

Pelo jeito as anotações são enviadas ao código inicialmente como `x y w h`, onde o `xy` são coordenadas em graus de 0 a 360 e o `w` a fov horizontal e `h` vertical. Posteriormente são convertidas??? Como enviar isso ao `tlts_kent` (código que converte bfov para kent)?
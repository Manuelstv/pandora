# Organização Pessoal

## Inputs

###todo

corrigir esse warning: "/home/mstveras/mmdetection_2/sphdet/bbox/kent_formator.py:486: ComplexWarning: Casting complex values to real discards the imaginary part K[1:,1:] = eigvects"

bbox_targets -> existem linhas (bonding kents) com valores de 0 em todas dims, pq isso acontece?

  logging.info("Filtering Parameters:")
    logging.info(f"IMAGE_LIMIT: {IMAGE_LIMIT}")
    logging.info(f"LIMIT: {LIMIT}")
    logging.info(f"MIN_BBOX_SIZE: {MIN_BBOX_SIZE}")
    logging.info(f"MIN_KAPPA: {MIN_KAPPA}")
    logging.info(f"BETA_RATIO_THRESHOLD: {BETA_RATIO_THRESHOLD}")

Adaptações Primeiro momento:
 - menos imagens
 - Bounding boxes muito pequenas -> KAPPA muito ALTO são excluidos
 - kappas muito altos ou muito baixos tambem sao removidos
 - kappa/beta  deve ser menor que TRESHOLD. é UMA CONDIÇÃao que esse treshold seja menor que 2, mas colocamos algo mais forte
 
 BRANCH ONLY kent loss: BRanch que será utilizada para regressão de BFoVs e converter para Kent somente para loss de acorod com porposta do claudio


- ler artigo e preparar apresentação.

- docstring em todas funcoes do kld

 - simplificar log/exp da KL

 - documentação mais extensiva do código.
 - - - github: substiruir branch sph kent pra virar main
 - - - incialização automatica do tmux
 - - - manter branch com line profiler.


viagem para floripa: quinta dia 5 de setembro

- ----- se ligar que eixtem diferenças no codigo quando roda com ou sem a gpu. Sem a gpu ta dando erro Nan no kld. Com a gpu nao da erro nennhum. tentar rodar semrpe com a gpu.


New 4th parameter = -1.0846471066844603 * x1 + -1.0843522573166815 *x2+ 126.34038381373834


New 4th parameter = -0.9806256706216581 * x1 + -0.6889740078642015 * x2 + 109.69413392828818


- > acredito que a causa do pq o codigo está tão lento com o sampling+moment testimation vem do facto que de ter usado vectorized operations para fazer o sampling inidividual. mas é implementado um loop para cada anotação o que é altamente ineficientE. o problema é que até onde entendo para ajeitar isso seria necessario ajeitar toda a parte de oment estimation o que vai consumir bastante trabalho.

 - > Na verdade tudo exceto as 3 linhas do kent moment estimation tao de acordo com o que precisaria ser feito com qualquer forma de moment estimation. a questao é fazer isso de forma vetorizada. De uma forma ou de outra esses calculos vao ter que ser feitos de maneira vetorizada na verdade  nao ser que consiga uma forma fechada para S e seja possivel calcular tudo na mao

ORDEM:

1 - artigo
2 - docstring   -- ------ ok
3 - github - criar nova branch apenas para loss  ---- ok
4 - simplificar log/exp da KL 
5 - tmux
6 - branch com line prifler



### Variáveis e Configurações
- **reg_decoded_bbox (bool)**: 
  - Se `True`, a regressão loss é aplicada diretamente em decoded bounding boxes, convertendo as caixas previstas e os alvos de regressão para o formato de coordenadas absolutas. 
  - DEFAULT: `False`. Deve ser `True` ao usar `IoULoss`, `GIoULoss` ou `DIoULoss` no bbox head.
  - **Nota**: Parece que essa variável deveria ser falsa.

### JSONs Importados
- **Formato**: Os JSONs importados pelo código (e.g., `instances_train.json`) estão em formato de `bfov`? E se estiverem no formato de bounding box?
- **NMS**: Precisa fazer algo? Inicialmente, pode ser omitido, mas como exatamente fazer isso?
- **bbox.sampler**: É relevante?

### Verificações
- **Arquivo**: Verificar se `bfov2kent_single_torch.py` está correto.
- **Linha 272**: seld.reg_decoded_box é usado na linha 272 de `anchor head.py`.
- **get_targets**: Onde é usado? Na loss. Onde a loss é usada?

### Código e Funções
- **Funções**: `bbox2delta` e `delta2bbox` precisam ser alteradas para fazer uma normalização que faça sentido para o caso das `kents`.
- **Classe**: Criar classe da loss, `iou` e `dataloader` no molde das classes de loss existentes.
- **Configuração**: Editar configuração para não computar `map`.

### Log de Atividades
- **Log (08/06)**:
  - Consegui rodar versão do código que converge com a branch (commit `b33fcbffbd980d09d7f494319f8d37523dde4c12`).
  - Problema com `pandora` nos commits seguintes (loss começava a dar `NaN` a partir de certo ponto).

### Papers e Leituras
- **Papers**:
  - "Can we trust bounding box annotations for object detection?"
  - "Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning".
  - `Rotated Object Detection with Circular Gaussian Distribution`
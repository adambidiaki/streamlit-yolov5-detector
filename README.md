# 🎯 Rastreamento de Objetos com Interface em Streamlit

Este projeto tem como objetivo realizar detecção e rastreamento de objetos em vídeo utilizando YOLO.  
A aplicação desenvolvida permite fazer upload de vídeos, aplicar detecção com diferentes níveis de confiança, selecionar classes personalizadas e visualizar métricas em tempo real como FPS, objetos rastreados e largura do frame.

A interface foi construída utilizando Streamlit, permitindo uma interação simples e intuitiva.

## Foi utilizado

- **Python**: 3.9.5
- **Ambiente**: Anaconda

## Executando o projeto

### 1️⃣ Criar o ambiente virtual com Anaconda

1. Abra o **Anaconda Prompt**.
   
2. Crie um novo ambiente virtual com o Python 3.9.5:

   ```bash
   conda create -n yolov5 python=3.9.5
   ```

3. Ative o ambiente:

   ```bash
   conda activate yolov5
   ```

### 2️⃣ Instalar as dependências

1. Navegue até o diretório onde o repositório foi clonado

2. Instale as dependências com o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```

### 3️⃣ Executar o projeto

1. Com o ambiente configurado e as dependências instaladas, você pode executar o projeto.

   ```bash
   python main.py
   ```

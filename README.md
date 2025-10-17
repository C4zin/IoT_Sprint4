# 🏍️ MotoTrack Vision — Gestão Inteligente de Frotas

## ⚙️ Link Pra Abrir o Projeto

https://iotsprint4-ntqudvchvkfsd5hgo5mvta.streamlit.app/

## 👥 Integrantes  
- **RM558317** - Cauã Sanches de Santana  
- **RM556511** - Angello Turano da Costa  
- **RM558576** - Leonardo Bianchi  

---

## 🚀 Projeto
O **MotoTrack** é uma solução inovadora que aplica **Visão Computacional** e **Inteligência Artificial** para apoiar a **gestão de frotas de motocicletas** em empresas de logística e delivery.  

O sistema combina **detecção de motos em vídeo (YOLOv8)** com **rastreamento em tempo real (ByteTrack)**, oferecendo:  
- 🎥 Processamento de vídeos enviados pelo usuário.  
- 📊 Métricas de motos detectadas e rastreadas em tempo real.  
- 📂 Exportação de resultados em **CSV** (logs) e **MP4** (vídeo anotado).   

---

## 📌 Tema
**“Visão Computacional para Gestão de Frotas de Motos”**  
> Projeto aplicado à empresa fictícia **MotoTrack**, especializada na locação e manutenção de motocicletas para serviços de entrega.

---

## ❗ Problema
Atualmente, a MotoTrack enfrenta dificuldades em seus pátios de armazenamento:  
- 📍 Falta de controle automatizado da localização e estado das motos.  
- 📝 Dependência de checklists manuais, sujeitos a **erros humanos**.  
- ⏳ Auditorias demoradas nos pátios.  
- ⚠️ Risco de divergência entre o cadastro digital e a realidade física.  

---

## 💡 Alternativas de Solução
- **Planilhas + QR Code**: baixo custo, mas ainda dependente da ação manual.  
- **Visão Computacional com IA (Escolhida ✅)**: solução escalável, confiável e automatizada para identificação e rastreamento da frota.  

---

## 🛠️ Tecnologias Utilizadas
- **[Python 3.13](https://www.python.org/)** 🐍  
- **[Streamlit](https://streamlit.io/)** 📊 → Dashboard interativo.  
- **[YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)** 🤖 → Detecção de motos.  
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** 🎯 → Rastreamento de múltiplos objetos.  
- **[OpenCV](https://opencv.org/)** 👁️ → Manipulação de imagens/vídeos.  
- **[Supervision](https://github.com/roboflow/supervision)** 🛡️ → Métricas e suporte ao rastreamento.  
- **Streamlit WebRTC (opcional)** 📡 → Webcam no navegador.  

---

## 📊 Funcionalidades

- **Upload de vídeo** 🎥  
  Suporta os formatos: `.mp4`, `.avi`, `.mov`, `.mkv`.

- **Detecção e rastreamento** 🏍️  
  Realiza detecção apenas de **motos** (classe COCO #3).

- **Métricas exibidas em tempo real**:  
  - ⏱️ **FPS estimado**  
  - 📍 **Motos no frame atual**  
  - 🆔 **IDs únicos de motos rastreadas**

- **Resultados exportáveis**:  
  - 📹 Vídeo com **anotações em MP4**  
  - 📑 Logs em **CSV** com frames, IDs e contagem de motos


---

## ⚙️ Instalação
Clone o repositório e instale as dependências:

```bash
git clone https://github.com/C4zin/IoT_Sprint4
cd IoT_Sprint4

# Crie ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windowss



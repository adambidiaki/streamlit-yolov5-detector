import numpy as np
import time
import torch
import cv2
import math
import cvzone
from sort import *
import os
from pathlib import Path

# Obtendo o diretório que o projeto está salvo
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Project root dir

# Diretório para salvar os videos
save_path = ROOT / "saved_videos"

# # Carregar modelo e alguns parâmetros não acessiveis ao usuário
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # Modelo pré-treinado da YOLO
# model.iou = 0.15  # NMS IoU threshold
# count = 0

# # Limite das coordenadas utilizadas para a contagem dos carros
# limits = [290, 705, 430, 705]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        # path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            # p = f"{path}{sep}{n}{suffix}"  # increment path
            p = f"{path}{sep}{n}"
            if not os.path.exists(p):
                break
        path = Path(p)

    return path


def detection(tfflie, confidence, save_video, enable_GPU, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stFrame):
    cap = cv2.VideoCapture(tfflie)

    # Habilitando/Desabilitando a GPU de acordo com a seleção dousuário
    device = torch.device('cuda' if torch.cuda.is_available() and enable_GPU else 'cpu')

    # Carreguando o modelo, ajustando o dispositivo selecionado
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    model.conf = confidence  # Alterando a confiança de detecção 
    model.iou = 0.15  # NMS IoU threshold       

    # Verificando em qual dispositivo o modelo está sendo executado
    device_name = next(model.parameters()).device.type
    print("Dispositivo: ", device_name)

    # Iniciando a variável de tempo que será utilizada para cálculo de FPS do vídeo
    previus_time = time.time()

    # Inicializa out com None
    out = None

    if save_video:
        # Verifica se já existe um diretório com onome padrão dentro da pasta, caso exista criar um novo
        save_dir = increment_path(Path(save_path) / "output", exist_ok=False)  # increment run
        (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        video_name = str(save_dir) + "/out.mp4"
        # print(video_name)

        # Obter as propriedades do vídeo
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Iniciando parâmetros para salvar videos
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(video_name, fourcc, video_fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret == True:
            # Cria uma cópia do frame para pós-processamento caso necessário
            img = frame.copy()

            # # Carregando o path da máscara
            # mask = cv2.imread("mask.png")
            # imgRegion = cv2.bitwise_and(img, mask)

            # Inferncia utilizando o modelo carregado 
            # results = model(imgRegion) # Imagem utilizada caso exista uma máscara
            results = model(img)
            detections = np.empty((0, 5))

            # # Linha utilizada para facilitar a observação dos objetos contabilizados
            # cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (0,0,255), 3)

            # Convertendo os resultados para um DataFrame pandas
            df_results = results.pandas().xyxy[0]            
            desierd_objects = df_results[df_results['class'].isin(assigned_class_id)] #Filtrando as classes desejadas
            num_objects = len(desierd_objects) # Contando o número de detecções deacordo com as classes selecionadas
            # print(df_results) # Exibindo as detecções

            for index, row in results.pandas().xyxy[0].iterrows():
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])

                d = (row['class'])

                # Confidence
                conf = math.ceil((row['confidence'] * 100)) / 100
        
                if d in assigned_class_id:
                    # cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                    rectx1, recty1 = ((x1+x2)/2, (y1+y2)/2)
                    rectcenter = int(rectx1), int(recty1)
                    cx = rectcenter[0]
                    cy = rectcenter[1]

                    # cv2.circle(img, (cx,cy), 3, (0,255,0), -1)
                    # cv2.putText(img, str(b), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

                resultsTracker = tracker.update(detections)

                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=0, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f'TrackId: {int(id)} ', (max(0, x1), max(35, y1)),
                                    scale=1.5, thickness=2, colorR=(17,152,255), offset=10)
                    
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                    # # Contabilização dos objetos detectados e atualização da linha
                    # if limits[1] - 15 < cy < limits[3] + 15:
                    #     if totalCount.count(id) == 0:
                    #         totalCount.append(id)
                    #         cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (0,255,0), 3)

            # Calculo do FPS atual, durante o processamento das imagens
            current_time = time.time()
            process_time = current_time - previus_time       
            fps = 1 / process_time
            previus_time = current_time

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
            # Salvar video com detecções e anotações
            if save_video:
                out.write(img)

            # Formatando os valores que serão exibidos da tela principal
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{'{:.1f}'.format(fps)}</h1>",
                            unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{(num_objects)}</h1>",
                            unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{(frame_width)}</h1>",
                            unsafe_allow_html=True)
            
            stFrame.image(img, channels='BGR', use_column_width=True)

        else:
            break

    # When everything done, release the video capture and video write objects 
    cap.release()
    if isinstance (out, cv2.VideoWriter): # new video
        out.release()  # release previous video writer

    # Closes all the frames 
    cv2.destroyAllWindows()

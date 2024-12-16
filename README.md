# KTL_project_15_CT_Cardiac_Segmentation

## Data Description
1. 학습용 데이터 (/Data/...)
   
            'LCTSC-Test-S1-201'
            'LCTSC-Test-S1-204'
            'LCTSC-Test-S1-103'
            'LCTSC-Test-S1-202'
            'LCTSC-Test-S1-203'
            'LCTSC-Test-S2-102'
            'LCTSC-Test-S2-104'
            'LCTSC-Test-S2-201'
            'LCTSC-Test-S2-103'
            'LCTSC-Test-S2-101'

   - 원본 CT 영상 (/Data/LCTSC-Test-##-###/###.nii)
   - 심장 분할 마스크 (/Data/LCTSC-Test-##-###/###-label.nii)
  
     
4. 성능 평가 데이터 (/Data/...)

            'LCTSC-Test-S1-101'
            'LCTSC-Test-S1-102'
            'LCTSC-Test-S1-104'
            'LCTSC-Test-S2-202'
            'LCTSC-Test-S2-203'
   
   - 원본 CT 영상 (/Data/LCTSC-Test-##-###/###.nii)
   - 심장 분할 마스크 (/Data/LCTSC-Test-##-###/###-label.nii)
## Prerequisites
Before you begin, ensure you have met the following requirements:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Other dependencies can be installed using `environment.yml`
  
## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ktlProject0/KTL_project_15_CT_Cardiac_Segmentation.git
cd KTL_project_15_CT_Cardiac_Segmentation
```
 - You can create a new Conda environment using `conda env create -f environment.yml`.

## Code Description
## Training.ipynb
  - 네트워크 학습 코드
## Evaluation.ipynb
  - 네트워크 성능 평가 및 심장 분할 결과 가시화
  - 학습완료 된 모델 가중치 (/Code/output/model_final.pth)
## model_torch.py
  - Residual UNet 아키텍쳐 빌드
## Preprocessing.py
  - Intensity windowing
  - 입력 이미지 해상도 조정
## _utils_torch.py
  - 네트워크를 구성에 필요한 부속 코드
  - Preprocessing 이후 데이터를 모델에 입력하기 위한 처리
## modules.torch.py
  - 네트워크를 구성에 필요한 부속 코드
## loss.py
  - 학습용 loss 함수 코드
  - Dice loss
  - Focal loss


# [참고] 3DSlicer에서 니프티(nii) 데이터 내보내기 가이드

## 1. 데이터 불러오기
1. 다운로드 받은 데이터의 각 폴더를 Slicer에 드래그 앤 드롭
![image01](https://github.com/user-attachments/assets/cd37a046-7c95-4825-9e26-3b908b4520b9)


2. 나타나는 창에서 OK 버튼 클릭
![image02](https://github.com/user-attachments/assets/59ce5c2c-03d3-425f-bb51-91d7448b82f2)

## 2. DICOM 데이터 추가  
1. Modules > Add DICOM Data 창으로 이동
2. 개체 별 데이터를 Load하여 데이터 확인

## 3. Heart 레이블맵 내보내기
1. Loaded data 창에서:
  - Heart를 제외한 다른 해부학적 구조물의 annotation 삭제
  - Heart 레이블 우클릭 후, Export labelmap 버튼 클릭
2. 생성된 라벨맵 객체 우클릭 후, Export to file 버튼 클릭

## 4. 파일 저장
1. Heart 레이블맵 저장:
  - Export format에서 nii 선택  
  - Export 버튼 클릭
  - 저장된 니프티 파일을 데이터셋으로 사용

2. CT 영상 저장:
  - 별도의 처리 과정 없이 객체 우클릭
  - Export to file 버튼을 통해 nii 형태로 저장  
  - 저장된 파일을 데이터셋으로 사용

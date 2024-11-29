# 15_Cardiac_Segmentation

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

# 原始 JSON 统一测试集指标

由 audit.py 自动生成；checkpoint 由验证集选取。run 名称保留原样，不能据目录名推断 epoch。

## 任务 7

| run | ap_epsilon_MAE | lat_epsilon_MAE | ap_epsilon_dec | lat_epsilon_dec |
|---|---:|---:|---:|---:|
| task07_keypoint_linear/dinov2 | 0.02822 | 0.03311 | 18.16378 | 18.55325 |
| task07_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/training_151424 | 0.03159 | 0.03520 | 20.16952 | 20.16907 |
| task07_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/training_229999 | 0.03301 | 0.03655 | 20.85395 | 20.81326 |
| task07_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/training_299999 | 0.03127 | 0.03620 | 19.95017 | 20.47558 |
| task07_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/training_378562 | 0.03142 | 0.03582 | 20.33105 | 20.41119 |
| task07_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/training_75712 | 0.03265 | 0.03561 | 20.73293 | 20.18624 |
| task07_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_113568 | 0.05666 | 0.06641 | 37.45348 | 40.25580 |
| task07_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_151424 | 0.04891 | 0.05387 | 32.25098 | 32.54232 |
| task07_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_37856 | 0.08330 | 0.10219 | 56.17091 | 59.85599 |
| task07_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_75712 | 0.06614 | 0.07843 | 43.69565 | 47.46080 |
| task07_keypoint_linear/dinov3 | 0.02230 | 0.02882 | 13.52123 | 15.94239 |
| task07_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_151424 | 0.01953 | 0.02503 | 12.33871 | 14.10337 |
| task07_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_227137 | 0.01962 | 0.02502 | 12.48028 | 14.17464 |
| task07_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_302849 | 0.01883 | 0.02513 | 11.65252 | 14.04757 |
| task07_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_378562 | 0.01914 | 0.02628 | 11.96633 | 14.75094 |
| task07_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_75712 | 0.01824 | 0.02650 | 11.90214 | 14.88788 |
| task07_keypoint_linear/rad_dino | 0.01792 | 0.02497 | 11.18225 | 14.18533 |
| task07_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_151424 | 0.01846 | 0.02352 | 11.39854 | 13.17816 |
| task07_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_227137 | 0.01879 | 0.02415 | 11.79989 | 13.45908 |
| task07_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_75712 | 0.01846 | 0.02397 | 11.43265 | 13.49680 |
| task07_keypoint_linear/rad_dino_maira2 | 0.01933 | 0.02568 | 11.77726 | 14.49195 |
| task07_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_151424 | 0.02053 | 0.02587 | 12.88903 | 14.38078 |
| task07_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_227137 | 0.02069 | 0.02658 | 13.02331 | 14.69159 |
| task07_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_299999 | 0.02025 | 0.02617 | 12.52863 | 14.56859 |
| task07_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_378562 | 0.02034 | 0.02619 | 12.54682 | 14.75178 |
| task07_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_75712 | 0.02001 | 0.02682 | 12.84593 | 14.96096 |
| task07_keypoint_linear/spine_dinov3_teacher_20k | 0.02192 | 0.02567 | 14.34661 | 14.52140 |
| task07_keypoint_linear/spine_rad_dino_maira2_teacher_20k | 0.01990 | 0.02550 | 12.17670 | 14.43928 |
| task07_keypoint_linear/spine_rad_dino_teacher_20k | 0.01835 | 0.02431 | 11.39775 | 13.77680 |
| task07_keypoint_lora/dinov2 | 0.01537 | 0.02196 | 9.96399 | 12.32150 |
| task07_keypoint_lora/dinov3 | 0.01399 | 0.02040 | 9.14192 | 11.51373 |
| task07_keypoint_lora/rad_dino | 0.01327 | 0.02067 | 8.56365 | 11.92830 |
| task07_keypoint_lora/rad_dino_maira2 | 0.01053 | 0.01821 | 6.58434 | 10.44487 |

## 任务 8

| run | MAE_mm | MRE_mm | SDR_2mm_percent | SDR_3mm_percent | SDR_4mm_percent | SDR_5mm_percent |
|---|---:|---:|---:|---:|---:|---:|
| task08_keypoint_linear/dinov2 | 19.10359 | 15.35078 | 2.99887 | 5.63938 | 9.26066 | 13.31573 |
| task08_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_151424 | 18.85509 | 15.25212 | 2.31988 | 5.54508 | 9.14749 | 13.46662 |
| task08_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_229999 | 18.06838 | 14.65295 | 2.67823 | 5.33761 | 9.16635 | 13.65522 |
| task08_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_299999 | 18.06527 | 14.59730 | 2.30102 | 5.37533 | 9.05319 | 13.99472 |
| task08_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_378562 | 18.48431 | 14.86595 | 2.71596 | 5.73369 | 9.65673 | 13.95700 |
| task08_keypoint_linear/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_75712 | 17.92013 | 14.62818 | 2.64051 | 5.46963 | 9.43040 | 14.20219 |
| task08_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_113568 | 32.22565 | 25.45129 | 0.58469 | 1.22595 | 2.35760 | 3.37608 |
| task08_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_37856 | 38.78819 | 30.38087 | 0.24519 | 0.52810 | 1.16937 | 2.03697 |
| task08_keypoint_linear/dinov2_vitb14_random_init_100pass_v2/training_75712 | 35.64447 | 28.07008 | 0.41494 | 0.92418 | 1.56545 | 2.48963 |
| task08_keypoint_linear/dinov3 | 16.12292 | 12.95358 | 3.60241 | 7.12939 | 12.29725 | 18.57790 |
| task08_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_151424 | 14.17011 | 11.53850 | 4.50773 | 9.78876 | 16.16371 | 23.44398 |
| task08_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_227137 | 14.12764 | 11.52517 | 5.62052 | 10.97699 | 17.46511 | 24.31158 |
| task08_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_302849 | 13.58369 | 11.02344 | 6.09204 | 11.75028 | 18.93625 | 25.95247 |
| task08_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_378562 | 14.00073 | 11.35196 | 5.43191 | 11.61826 | 18.18182 | 24.99057 |
| task08_keypoint_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_75712 | 14.48872 | 11.81534 | 5.14900 | 10.35458 | 17.61599 | 24.83968 |
| task08_keypoint_linear/rad_dino | 16.57647 | 13.38342 | 3.79102 | 7.82724 | 12.67446 | 18.97397 |
| task08_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_151424 | 12.37503 | 9.92270 | 5.82799 | 12.20294 | 19.91701 | 28.02716 |
| task08_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_227137 | 11.90807 | 9.54590 | 6.39381 | 13.41003 | 21.36929 | 30.47906 |
| task08_keypoint_linear/rad_dino_init_rad_crops_100pass_plan/training_75712 | 12.49679 | 10.03041 | 6.05432 | 12.20294 | 19.06828 | 27.51792 |
| task08_keypoint_linear/rad_dino_maira2 | 15.07430 | 12.18495 | 4.26254 | 9.03433 | 14.65485 | 21.93512 |
| task08_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/teacher_training_151424 | 11.67354 | 9.33871 | 7.48774 | 15.03206 | 23.65145 | 33.62882 |
| task08_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/teacher_training_227137 | 10.94950 | 8.73815 | 9.29838 | 19.10600 | 29.02678 | 39.47567 |
| task08_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/teacher_training_299999 | 10.24375 | 8.18208 | 11.31648 | 21.89740 | 33.74198 | 43.73821 |
| task08_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/teacher_training_378562 | 9.15216 | 7.27322 | 13.56092 | 26.25424 | 39.96605 | 51.39570 |
| task08_keypoint_linear/rad_dino_maira2_init_rad_crops_100pass_plan/teacher_training_75712 | 12.50548 | 10.04332 | 6.39381 | 13.42889 | 21.01094 | 28.81931 |
| task08_keypoint_linear/spine_dinov3_teacher_20k | 14.24096 | 11.65638 | 5.22444 | 10.61863 | 17.44625 | 24.50019 |
| task08_keypoint_linear/spine_rad_dino_maira2_teacher_20k | 13.60727 | 11.07029 | 5.77141 | 12.12750 | 19.70954 | 27.49906 |
| task08_keypoint_linear/spine_rad_dino_teacher_20k | 14.55198 | 11.78135 | 5.46963 | 11.01471 | 17.67258 | 24.33044 |
| task08_keypoint_lora/dinov2 | 7.16308 | 5.78896 | 34.30781 | 52.33874 | 64.37194 | 71.76537 |
| task08_keypoint_lora/dinov3 | 7.63258 | 6.16938 | 26.44285 | 46.09581 | 59.73218 | 70.14334 |
| task08_keypoint_lora/rad_dino | 7.27726 | 5.83852 | 26.29197 | 44.85100 | 59.31724 | 69.59638 |
| task08_keypoint_lora/rad_dino_maira2 | 5.85514 | 4.69482 | 47.24632 | 68.57790 | 80.04527 | 86.13731 |

## 任务 9

| run | IDR | IRA |
|---|---:|---:|
| task09_linear/dinov2 | 76.41434 | 23.18841 |
| task09_linear/dinov2_100ep | 79.84064 | 38.40580 |
| task09_linear/dinov2_vitb14_lvd142m_100pass_plan/training_151424 | 67.09163 | 19.56522 |
| task09_linear/dinov2_vitb14_lvd142m_100pass_plan/training_229999 | 67.64940 | 21.01449 |
| task09_linear/dinov2_vitb14_lvd142m_100pass_plan/training_299999 | 67.01195 | 21.01449 |
| task09_linear/dinov2_vitb14_lvd142m_100pass_plan/training_378562 | 67.33068 | 18.84058 |
| task09_linear/dinov2_vitb14_lvd142m_100pass_plan/training_75712 | 67.56972 | 23.18841 |
| task09_linear/dinov2_vitb14_random_init_100pass_v2/training_113568 | 63.66534 | 12.31884 |
| task09_linear/dinov2_vitb14_random_init_100pass_v2/training_37856 | 49.72112 | 2.17391 |
| task09_linear/dinov2_vitb14_random_init_100pass_v2/training_75712 | 58.40637 | 5.79710 |
| task09_linear/dinov3 | 84.62151 | 47.10145 |
| task09_linear/dinov3_100ep | 84.54183 | 46.37681 |
| task09_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_151424 | 85.57769 | 58.69565 |
| task09_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_227137 | 84.94024 | 57.97101 |
| task09_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_302849 | 87.17131 | 59.42029 |
| task09_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_378562 | 87.17131 | 57.24638 |
| task09_linear/dinov3_vitb16_lvd1689m_100pass_plan/training_75712 | 82.86853 | 51.44928 |
| task09_linear/rad_dino | 86.77291 | 57.24638 |
| task09_linear/rad_dino_100ep | 86.45418 | 50.72464 |
| task09_linear/rad_dino_init_rad_crops_100pass_plan/training_151424 | 87.49004 | 62.31884 |
| task09_linear/rad_dino_init_rad_crops_100pass_plan/training_227137 | 87.49004 | 63.76812 |
| task09_linear/rad_dino_init_rad_crops_100pass_plan/training_75712 | 87.80876 | 60.86957 |
| task09_linear/rad_dino_maira2 | 88.84462 | 55.79710 |
| task09_linear/rad_dino_maira2_100ep | 85.97610 | 45.65217 |
| task09_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_151424 | 87.09163 | 55.79710 |
| task09_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_227137 | 87.33068 | 64.49275 |
| task09_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_299999 | 87.80876 | 64.49275 |
| task09_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_378562 | 88.20717 | 64.49275 |
| task09_linear/rad_dino_maira2_init_rad_crops_100pass_plan/training_75712 | 85.33865 | 52.17391 |
| task09_linear/spine_dinov3_teacher_20k | 85.65737 | 57.97101 |
| task09_linear/spine_rad_dino_maira2_teacher_20k | 84.86056 | 52.17391 |
| task09_linear/spine_rad_dino_teacher_20k | 87.64940 | 52.89855 |
| task09_lora/dinov2 | 94.82072 | 81.15942 |
| task09_lora/dinov3 | 90.59761 | 76.08696 |
| task09_lora/rad_dino | 92.58964 | 75.36232 |
| task09_lora/rad_dino_maira2 | 95.53785 | 86.23188 |

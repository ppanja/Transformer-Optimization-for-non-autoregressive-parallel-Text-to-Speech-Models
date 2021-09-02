import glob
import os

#list_of_files = glob.glob('/mnt/fast0/pp837/project/files/NewDummyTxt/*.txt')           # create the list of file
FI = open('/mnt/fast0/pp837/project/files/ljs_audio_text_filelist.txt', 'r') 
FTR = open('/mnt/fast0/pp837/project/files/ljs_audio_text_train_filelist.txt', 'w') 
FT = open('/mnt/fast0/pp837/project/files/ljs_audio_text_test_filelist.txt', 'w') 
FV = open('/mnt/fast0/pp837/project/files/ljs_audio_text_val_filelist.txt', 'w') 

i=0

for line in FI:
    if i <= 999:
        FTR.write(line)
        i = i + 1
    elif (i > 999 and i <= 1064):
        FV.write(line)
        i = i + 1
    elif i > 1064:
        FT.write(line)
        i = i + 1

FI.close()
FTR.close()
FT.close()
FV.close()

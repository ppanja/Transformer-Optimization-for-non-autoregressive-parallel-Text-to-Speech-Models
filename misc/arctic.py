import glob
import os
import re

#list_of_files = glob.glob('/mnt/fast0/pp837/project/files/NewDummyTxt/*.txt')           # create the list of file

FR = open('/mnt/fast0/pp837/project/files/txt.done.data', 'r') 

FO = open('/mnt/fast0/pp837/project/files/ljs_audio_text_filelist.txt', 'w') 
for line in FR:
  s = re.sub(r'^\(|"|\)$', '', line).split()
  s.insert(1,'.wav')
  s.insert(2,'|')
  l = len(s)
  fs = "DUMMY/"+"".join(s[0:3])+" ".join(s[3:l])+"\n"
  #print(fs)
  FO.write(fs)

FR.close()
FO.close()

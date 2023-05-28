
import time
from getkeys import key_check
import cv2
from grabscreen import grab_screen
def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused


window_size = (640,220,1280,620)#384,352  192,176 96,88 48,44 24,22
# station window_size

blood_window = (60,91,280,562)
# used to get boss and self blood

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

paused = True
# used to stop training
if __name__ == '__main__':
    i = 0






























    while True:

        # screen_reshape = cv2.resize(screen_gray,(96,86))

        screen_gray = grab_screen(window_size)  # 灰度图像收集
        # cv2.imwrite("runs/"+str(i) + ".jpg", screen_gray)
        # cv2.imshow('window3',printscreen)gggggggggggg
        # cv2.imshow('window2',screen_reshape)

        print("imgsz", imgsz)




        last_time = time.time()
        print(i)
        keys = key_check()
        i=i+1


        paused = pause_game(paused)
        if 'G' in keys:
            print('stop testing DQN')
            break

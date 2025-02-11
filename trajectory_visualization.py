import arcle
import action_sequence_parser
import json
import numpy as np
import time

prob_id = '3d9a5b9b'
ansi256arc = [0,12,9,10,11,8,13,208,14,52] # ANSI Color Code

def render_ansi(grid, sel, clip, action):

    
    H=30
    W=30

    grid_dim= (len(grid), len(grid[0]))
    for i in range(H):
        for j in range(W):
            st = "[]" if sel[i,j] else "  " 

            if i >= grid_dim[0] or j>= grid_dim[1]:
                print(f'\033[47m{st}', end='')
            else:
                print("\033[48;5;"+str(ansi256arc[grid[i,j]])+f"m{st}", end='')

        print("\033[0m  ",end='')

        print('\033[0m')

    print('Dimension : '+ str(grid_dim), end=' ')
    print('\033[KNextAction : ' + action + '\033[K')

if __name__=="__main__":
    df = action_sequence_parser.preprocess()
    auto_mode = 0
    interval =0.5
    inthemenu=False
    while True:
        if not inthemenu:
            prob_id =input(f"{'AUTOMODE '+str(interval)+' / ' if auto_mode else ''}Press task number (1~400) :")
            try:
                prob_id=int(prob_id)
            except:
                if prob_id[0] == 'a':

                    if len(prob_id)>1 and prob_id[1:].replace('.', '', 1).isdigit():
                        auto_mode=1
                        interval = float(prob_id[1:])
                    if prob_id[1:] == '':
                        auto_mode= 1-auto_mode
                else:
                    print('> invalid number')
                continue
            if prob_id <= 0 or prob_id > 400:
                print('> invalid problem')
                continue
        inthemenu=True
        probdf= df[df['taskId']==prob_id]
        print(f'{len(probdf)} records found for problem {prob_id}.')
        print('Index\ts\t  log\t  uId\t   score\ttry')
        print('----------------------------------------------------')
        for i in range(len(probdf)):
            pp = probdf.iloc[i]
            succstr = '\033[0;42mO\033[0m' if pp['success'] else '\033[0;41mX\033[0m'
            print(f'{i+1:3}\t{succstr:4}\t{pp["logId"]:5}\t{pp["userId"]:5}\t{pp["score"]:8}\t{pp["trial"]:3}')
        trajind = input('Type index to view the log (q to go back): ')
        if trajind == 'q':
            inthemenu=False
            continue
        try:   
            trajind = int(trajind)
            if trajind == 0 or trajind > len(probdf):
                print('> invalid number')
                continue
        except:
            print('> invalid number')
            continue
        traj= json.loads(probdf.iloc[trajind-1]['actionSequence'])
        
        print(f'>>> Go to https://o2arc.com/task/{prob_id} to see the problem.\n')

        cursor = 0
        adder = +1
        try:
            while True:
                act=traj[cursor]
                print(f'Problem {prob_id:03} / Trajectory {trajind:3}')
                sel = np.zeros((30,30),dtype=int)
                op = act['operation']
                grid = np.array(act['grid'])
                h, w = grid.shape
                for aaa in act['object']:
                    sel[aaa['y'],aaa['x']]=1
                    if 0<=aaa['y']<h and 0<=aaa['x']<w:
                        grid[aaa['y'],aaa['x']]=aaa['color']
                render_ansi(grid, sel, None,  act['operation'])
                print(f'Index {cursor+1}/{len(traj)} {"forward" if adder>0 else "backward"} mode')
                if not auto_mode:
                    stop = input('\033[K')
                else:
                    stop=''
                    print()
                print(f'\033[K\033[{1}A', end='')
                if auto_mode:
                    time.sleep(interval)
                    if cursor==len(traj)-1:
                        input('End of trajectory. Press any key to continue.')
                        break
                if stop == 'q':
                    break
                elif stop=='':
                    cursor+=adder
                    if cursor >= len(traj):
                        cursor=len(traj)-1
                    elif cursor < 0:
                        cursor=0
                elif stop=='f':
                    adder=+1
                elif stop=='b':
                    adder=-1
                
                
                print(f'\033[{33}A\033[K', end='')
        except KeyboardInterrupt:
            print('Interrupted')
            continue
        except Exception as e:
            print('Error occured in parsing next state: ', e)
            continue
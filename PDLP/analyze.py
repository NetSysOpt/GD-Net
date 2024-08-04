import os
import json
import gzip

mode = 'covering_covering_5000_5000_20.0'
mode = 'covering_covering_10000_10000_5.0'
mode = 'model3_lp_1000_1000_600.0x0'
mode = 'model3_lp_1000_1000_600.0x0'
mode = 'model3_lp_10000_10000_5.0x0'
mode = 'model3_lp_5000_5000_20.0x0'

tar = f'../logs/test_log_{mode}dchannel_feat5.log'

preds= open(tar,'r')

def read_json(fnm):
    # Opening JSON file
    f = gzip.open(fnm,'r')
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data

res_file = open(f'./res_{mode}.csv','w')
st=f'fnm itn time info pred\n'
res_file.write(st)

total_time = 0.0
total_ins = 0
for line in preds:
    if 'Instance' not in line:
        continue
    lst= line.split(' ')
    lst=[x for x in lst if x!='']
    print(line)
    print(lst)
    fnm = lst[1].replace(':','')
    pred = float(lst[5].split(':')[-1])
    print(fnm,pred)
    iix = int(fnm.split('_')[-1].replace('.pkl',''))
    
    log_file = f'./logs/test_{iix}_full_log.json.gz'
    
    print(log_file)
    res = read_json(log_file)
    time = -1
    # for rr in res['iteration_stats']:
    #     print(rr)
    #     input()
    for rr in res['iteration_stats']:
        time = float(rr['cumulative_time_sec'])
        itn = int(rr['iteration_number'])
        info = rr['convergence_information'][0]['primal_objective']
        info = float(info)
        if info < 0:
            info = -info
            
            if info>=pred:
                print('NEG',itn,time,info,pred)
                st=f'{fnm} {itn} {time} {info} {pred}\n'
                res_file.write(st)
                break
            # else:
            #     print('XXX',itn,time,info,pred)
        elif info > 0:
            
            if info<=pred:
                print(itn,time,info,pred)
                st=f'{fnm} {itn} {time} {info} {pred}\n'
                res_file.write(st)
                break
            # else:
            #     print('XXX',itn,time,info,pred)
    if time > 0:
        total_time+=time
    total_ins+=1
total_time = total_time/total_ins
print(total_time,total_ins)
st=f'{total_time} {total_ins}\n'
res_file.write(st)

res_file.close()
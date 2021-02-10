import numpy as np
import csv
import pandas as pd

data_training = pd.read_csv('data/UNSW_NB15_training-set_multi.csv',encoding='utf-8',low_memory=False
                 )
data_testing = pd.read_csv('data/UNSW_NB15_testing-set_multi.csv',encoding='utf-8',low_memory=False
                 )
#data1['state']=data1['state'].replace(['CON','ECO','FIN','INT','no','PAR','REQ','RST','URN'],[1,2,3,4,5,6,7,8,9])
#data1['service']=data1['service'].replace(['-','dhcp','dns','ftp','ftp-data','http','irc','pop3','radius','smtp','snmp','ssh','ssl'],[1,2,3,4,5,6,7,8,9,10,11,12,13])
# #data1['proto']=data1['proto'].replace(['3pc','a/n','aes-sa3-d','any','argus','aris','arp','ax.25','bbn-rcc','bna','br-sat-mon','cbt','cftp','chaos','compaq-peer','cphb',
#                                        'cpnx','crtp','crudp','dcn','ddp','ddx','dgp','egp','eigrp','emcon','encap','etherip','fc','fire','ggp','gmtp','gre','hmp','i-nlsp','iatp','ib',
#                                        'icmp','idpr','idpr-cmtp','idrp','ifmp','igmp','igp','il','ip','ipcomp','ipcv','ipip','iplt','ipnip','ippc','ipv6','ipv6-frag','ipv6-no',
#                                        'ipv6-opts','ipv6-route','ipx-n-ip','irtp','isis','iso-ip','iso-tp4','kryptolan','l2tp','larp','leaf-1','leaf-2','merit-inp','mfe-nsp','mhrp',
#                                        'micp','mobile','mtp','mux','narp','netblt','nsfnet-igp','nvp','ospf','pgm','pim','pipe','pnni','pri-enc','prm','ptp','pup','pvp','qnx','rdp',
#                                        'rsvp','rtp','rvd','sat-expak','sat-mon','sccopmce','scps','sctp','sdrp','secure-vmtp','sep','skip','sm','smp','snp','sprite-rpc','sps','srp',
#                                        'st2','stp','sun-nd','swipe','tcf','tcp','tlsp','tp++','trunk-1','trunk-2','ttp','udp','unas','uti','vines','visa','vmtp','vrrp','wb-expak','wb-mon',
#                                        'wsn','xnet','xns-idp','xtp','zero'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
#                                                                              41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
#                                                                              80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
#                                                                              115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133])
data_training['multi_label']=data_training['multi_label'].replace(['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers',"Generic",'Reconnaissance','Shellcode','Worms'],[0,1,2,3,4,5,6,7,8,9])
data_training.to_csv('data/UNSW_NB15_training-set_multi.csv',index=False, encoding='utf-8')
print('data is finished')
# #data2['state']=data2['state'].replace(['CON','ECO','FIN','INT','no','PAR','REQ','RST','URN'],[1,2,3,4,5,6,7,8,9])
# #data2['service']=data2['service'].replace(['-','dhcp','dns','ftp','ftp-data','http','irc','pop3','radius','smtp','snmp','ssh','ssl'],[1,2,3,4,5,6,7,8,9,10,11,12,13])
# #data2['proto']=data2['proto'].replace(['3pc','a/n','aes-sa3-d','any','argus','aris','arp','ax.25','bbn-rcc','bna','br-sat-mon','cbt','cftp','chaos','compaq-peer','cphb',
#                                        'cpnx','crtp','crudp','dcn','ddp','ddx','dgp','egp','eigrp','emcon','encap','etherip','fc','fire','ggp','gmtp','gre','hmp','i-nlsp','iatp','ib',
#                                        'icmp','idpr','idpr-cmtp','idrp','ifmp','igmp','igp','il','ip','ipcomp','ipcv','ipip','iplt','ipnip','ippc','ipv6','ipv6-frag','ipv6-no',
#                                        'ipv6-opts','ipv6-route','ipx-n-ip','irtp','isis','iso-ip','iso-tp4','kryptolan','l2tp','larp','leaf-1','leaf-2','merit-inp','mfe-nsp','mhrp',
#                                        'micp','mobile','mtp','mux','narp','netblt','nsfnet-igp','nvp','ospf','pgm','pim','pipe','pnni','pri-enc','prm','ptp','pup','pvp','qnx','rdp',
#                                        'rsvp','rtp','rvd','sat-expak','sat-mon','sccopmce','scps','sctp','sdrp','secure-vmtp','sep','skip','sm','smp','snp','sprite-rpc','sps','srp',
#                                        'st2','stp','sun-nd','swipe','tcf','tcp','tlsp','tp++','trunk-1','trunk-2','ttp','udp','unas','uti','vines','visa','vmtp','vrrp','wb-expak','wb-mon',
#                                        'wsn','xnet','xns-idp','xtp','zero'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
#                                                                              41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
#                                                                              80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
#                                                                              115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133])
data_testing['multi_label']=data_testing['multi_label'].replace(['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers',"Generic",'Reconnaissance','Shellcode','Worms'],[0,1,2,3,4,5,6,7,8,9])
# data2['state']=data2['state'].replace(['ACC','CLO'],[5,9])
data_testing.to_csv('data/UNSW_NB15_testing-set_multi.csv',index=False, encoding='utf-8')
print('data is finished')


#print (y)


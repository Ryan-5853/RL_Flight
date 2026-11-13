from pathlib import Path
import struct,re,json,collections
import numpy as np
SOURCE=Path('/home/ryan/RL_Flight/log/26.11.4/log_0_2026-11-4-15-33-10.ulg')
OUT=Path(__file__).resolve().parent
fmt={};subs={};records=collections.defaultdict(list);params={};info={};logs=[];counts=collections.Counter();dropouts=[]
b=SOURCE.read_bytes();assert b[:7]==b'ULog\x01\x12\x35';start=struct.unpack_from('<Q',b,8)[0];pos=16
primitive={'uint64_t':'<u8','int64_t':'<i8','uint32_t':'<u4','int32_t':'<i4','uint16_t':'<u2','int16_t':'<i2','uint8_t':'u1','int8_t':'i1','bool':'?','float':'<f4','double':'<f8','char':'S1'}
cache={}
def dtype(name):
 if name in cache:return cache[name]
 fields=[]
 for f in fmt[name].split(';'):
  if not f:continue
  typ,key=f.split();m=re.fullmatch(r'(\w+)(?:\[(\d+)\])?',typ);base,n=m.groups();d=primitive[base] if base in primitive else dtype(base)
  fields.append((key,d,(int(n),)) if n else (key,d))
 cache[name]=np.dtype(fields);return cache[name]
def kv(payload):
 n=payload[0];key=payload[1:1+n].decode();typ,name=key.split(' ',1);value=payload[1+n:]
 if typ in primitive:return name,np.frombuffer(value,dtype=primitive[typ])[0].item()
 return name,value.rstrip(b'\0').decode(errors='replace')
while pos+3<=len(b):
 size,kind=struct.unpack_from('<HB',b,pos);pos+=3;p=b[pos:pos+size];pos+=size
 if len(p)<size:print('truncated tail',pos-size);break
 k=chr(kind);counts[k]+=1
 if k=='F':name,fields=p.decode().split(':',1);fmt[name]=fields
 elif k=='A':multi,mid=struct.unpack_from('<BH',p);subs[mid]=(p[3:].decode(),multi)
 elif k=='D':mid=struct.unpack_from('<H',p)[0];records[subs[mid]].append(p[2:])
 elif k=='P':name,val=kv(p);params[name]=val
 elif k=='I':name,val=kv(p);info[name]=val
 elif k=='L':logs.append({'level':p[0],'timestamp':struct.unpack_from('<Q',p,1)[0],'text':p[9:].decode(errors='replace')})
 elif k=='O':dropouts.append(struct.unpack('<H',p)[0])
arrays={};summary={}
for (name,multi),rec in records.items():
 d=dtype(name);sz=len(rec[0]);assert all(len(x)==sz for x in rec)
 # Trailing padding bytes may be omitted from the wire.
 assert sz<=d.itemsize,(name,sz,d.itemsize)
 arr=np.frombuffer(b''.join(x+b'\0'*(d.itemsize-len(x)) for x in rec),dtype=d).copy();key=f'{name}_{multi}';arrays[key]=arr
 summary[key]={'rows':len(arr),'first_s':float(arr['timestamp'][0]/1e6),'last_s':float(arr['timestamp'][-1]/1e6),'fields':list(d.names),'format':fmt[name]}
np.savez_compressed(OUT/'topics.npz',**arrays)
(OUT/'metadata.json').write_text(json.dumps({'source':str(SOURCE),'start_us':start,'counts':dict(counts),'dropouts_ms':dropouts,'info':info,'params':params,'logs':logs,'topics':summary},indent=2))
print(json.dumps({'start':start,'dropouts':dropouts,'info':info,'topics':summary,'logs':logs},indent=2))

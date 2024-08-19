
import json

class Database:

    def insert(self,n,e,p):
        with open('users.json','r') as f:
            data = json.load(f)
            f.close()
            if e in data:
                return 0
            else:
                data[e] = [n,p]

        with open('users.json','w') as f:
            json.dump(data,f,indent=4)
            f.close()
            return 1

    def search(self,e,p):
        with open('users.json','r') as f:
            data = json.load(f)
            if e in data:
                if data[e][1] == p:
                    return 1
                else:
                    return 0
            else:
                return 0
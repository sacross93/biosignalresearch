# -*- coding: utf-8 -*-
import datetime
import struct
import gzip


#2017-11-13 최종 수정
#check_data() 함수와 read_data_sample()함수 추가
#pleth가 들어왔는지 판별 가능
#정확도는 breakcnt/totalcnt로 변경할 수 있음.

class vital_reader(object):
    def __init__(self,file):
        self.vital_file = gzip.open(file,"rb")
        self.vital_file.seek(0)
        self.sign = self.vital_file.read(4)
        self.format_ver = int(struct.unpack('<I', self.vital_file.read(4))[0])
        self.headerlen = int(struct.unpack('<H', self.vital_file.read(2))[0])
        self.tzbias = int(struct.unpack('<H', self.vital_file.read(2))[0])
        self.inst_id = int(struct.unpack('<I', self.vital_file.read(4))[0])
        self.prog_ver = int(struct.unpack('<I', self.vital_file.read(4))[0])
        self.offset = 20
        self.pflag = [0,0,0] #pleth 확인
        self.cflag = 0 #pleth의 확인 flag


    def print_header(self):
        print("sign :", self.sign)
        print("format_ver :", self.format_ver)
        print("headerlen :", self.headerlen)
        print("tzbias :", self.tzbias)
        print("inst_id :", self.inst_id)
        print("prog_ver :", self.prog_ver)

    def read(self):
        while(1):
            self.vital_file.seek(self.offset)
            self.type = int(struct.unpack('<B', self.vital_file.read(1))[0])
            self.datalen = int(struct.unpack('<I', self.vital_file.read(4))[0])
            self.offset += 5 + self.datalen

            if self.type == 9: #devinfo
                self.did = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.typenamelen = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.typename = self.vital_file.read(self.typenamelen)
                self.devnamelen = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.devname = self.vital_file.read(self.devnamelen)
                self.portlen = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.port = self.vital_file.read(self.portlen)

            if self.type == 0: #trkinfo
                self.tid = int(struct.unpack('<H', self.vital_file.read(2))[0])
                self.rec_type = int(struct.unpack('<B', self.vital_file.read(1))[0])
                self.rec_fmt = int(struct.unpack('<B', self.vital_file.read(1))[0])
                self.namelen = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.name = self.vital_file.read(self.namelen)
                self.unitlen = int(struct.unpack('<I', self.vital_file.read(4))[0])
                self.unit = self.vital_file.read(self.unitlen)
                self.minval = struct.unpack('<f', self.vital_file.read(4))[0]
                self.maxval = struct.unpack('<f', self.vital_file.read(4))[0]
                self.color = self.vital_file.read(4)
                self.strate = struct.unpack('<f', self.vital_file.read(4))[0]
                self.adc_gain = struct.unpack('<d', self.vital_file.read(8))[0]
                self.adc_offset = struct.unpack('<d', self.vital_file.read(8))[0]
                self.montype = int(struct.unpack('<B', self.vital_file.read(1))[0])
                self.did = int(struct.unpack('<I', self.vital_file.read(4))[0])

            if self.type == 6: #cmd
                self.cmd = int(struct.unpack('<B', self.vital_file.read(1))[0]) # 5,6이 아니면 패스
                self.cnt = int(struct.unpack('<H', self.vital_file.read(2))[0])  # 트랙의 갯수
                self.tids = int(struct.unpack('<H', self.vital_file.read(2))[0]) # 배열식인것 같음. 수정필요. 현재 1개

            if self.type == 1: #rec
                self.infolen = int(struct.unpack('<H', self.vital_file.read(2))[0])
                self.dt = struct.unpack('<d', self.vital_file.read(8))[0]
                self.tid = int(struct.unpack('<H', self.vital_file.read(2))[0])
                break


    def print_devinfo(self):
        try:
            print("did :" ,self.did)
            print("typename :",self.typename)
            print("devname :",self.devname)
            print("port :", self.port)
        except:
            print("you need to read first")

    def print_trkinfo(self):
        try:
            print("tid :",self.tid)
            print("rec_type :",self.rec_type)
            print("rec_fmt :",self.rec_fmt)
            print("name :",self.name)
            print("unit :",self.unit)
            print("minval :",self.minval)
            print("maxval :", self.maxval)
            print("color :",self.color)
            print("strate :",self.strate)
            print("adc_gain :",self.adc_gain)
            print("adc_offset :",self.adc_offset)
            print("montype :",self.montype)
            print("did :",self.did)
        except:
            print("you need to read first")

    def print_cmd(self):
        try:
            print("cmd :",self.cmd)
            print("cnt :",self.cnt)
            print("tids :",self.tids)
        except:
            print("you need to read first")

    def print_rec(self):
        try:
            print("dt :",self.dt)
            print("tid :",self.tid)
        except:
            print("you need to read first")

    def read_data(self):
        try:
            data = []
            templen = (self.datalen - 16) / 2
            if self.rec_type == 2:
                data = int(struct.unpack('<I', self.vital_file.read(4))[0])
            elif self.rec_type == 1:
                self.num = int(struct.unpack('<I', self.vital_file.read(4))[0])

                if self.rec_fmt == 1 :
                    for i in range(1, self.num):
                        data.append(struct.unpack('<f', self.vital_file.read(4))[0])

                elif self.rec_fmt == 2 :
                    for i in range(1, self.num):
                        data.append(struct.unpack('<d', self.vital_file.read(4))[0])

                elif self.rec_fmt == 6 :
                    for i in range(1, int(templen)):
                        data.append(int(struct.unpack('<h', self.vital_file.read(2))[0]))

                else :
                    for i in range(1, int(templen)):
                        data.append(int(struct.unpack('<I', self.vital_file.read(4))[0]))

            return data

        except:
            print("you need to read first")



    def read_data_sample(self):
        try:
            data = []

            if self.datalen-16 ==0:
                return

            #templen = (self.datalen - 16) / 2
            templen = 6
            if self.rec_type == 2:
                data = int(struct.unpack('<I', self.vital_file.read(4))[0])
            elif self.rec_type == 1:
                self.num = int(struct.unpack('<I', self.vital_file.read(4))[0])

                if self.rec_fmt == 1 :
                    for i in range(1, self.num):
                        data.append(struct.unpack('<f', self.vital_file.read(4))[0])

                elif self.rec_fmt == 2 :
                    for i in range(1, self.num):
                        data.append(struct.unpack('<d', self.vital_file.read(4))[0])

                elif self.rec_fmt == 6 :
                    for i in range(1, int(templen)):
                        data.append(int(struct.unpack('<h', self.vital_file.read(2))[0]))

                else :
                    for i in range(1, int(templen)):
                        data.append(int(struct.unpack('<I', self.vital_file.read(4))[0]))

            return data

        except:
            print("you need to read first")

    # pleth가 들어왔는지 판별 가능
    # 정확도는 breakcnt/totalcnt로 변경할 수 있음.


    def check_data(self):
        try:
            totalcnt = 0
            breakcnt =0
            for i in range(10000000):
                self.read()

                if (self.name == b'PLETH' ):
                    totalcnt += 1
                    self.pflag[0] = 1
                    wavedata = self.read_data_sample()

                    if self.datalen -16 !=0:
                        if wavedata[0] == 0 or (wavedata[0]+wavedata[1]+wavedata[2]+wavedata[3]+wavedata[4]+wavedata[5]) % wavedata[5] ==0 :
                            breakcnt +=1


        except:
            if self.pflag[0] == 0 or breakcnt/totalcnt>0.4:
                print("PLETH, ECG is not correct data")
                #print(breakcnt / totalcnt)
            else:
                print("PLETH, ECG is correct data")
                print(breakcnt/totalcnt)

                self.cflag = 1
            pass


"""
    def print_packetheader(self, num, filename):

        excel = win32com.client.Dispatch("Excel.Application")
        wb = excel.Workbooks.Add()
        ws = wb.Worksheets("Sheet1")

        ws.Cells(1, 1).Value = ("did")
        ws.Cells(1, 2).Value = ("typename")
        ws.Cells(1, 3).Value = ("devname" )
        ws.Cells(1, 4).Value = ("port")
        ws.Cells(1, 5).Value = ("tid")
        ws.Cells(1, 6).Value = ("rec_type")
        ws.Cells(1, 7).Value = ("rec_fmt")
        ws.Cells(1, 8).Value = ("name")
        ws.Cells(1, 9).Value = ("unit")
        ws.Cells(1, 10).Value = ("minval")
        ws.Cells(1, 11).Value = ("maxval")
        ws.Cells(1, 12).Value = ("color")
        ws.Cells(1, 13).Value = ("strate")
        ws.Cells(1, 14).Value = ("adc_gain")
        ws.Cells(1, 15).Value = ("adc_offset")
        ws.Cells(1, 16).Value = ("montype")
        ws.Cells(1, 17).Value = ("did")
        ws.Cells(1, 18).Value = ("dt")
        ws.Cells(1, 19).Value = ("tid")

        for i in range(num):

            if i == 10000:
                print("10000번째")
            if i == 20000:
                print("20000번째")
            if i == 30000:
                print("30000번째")
            if i == 40000:
                print("40000번째")
            if i == 50000:
                print("50000번째")


            self.read()
            ws.Cells(i+2, 1).Value = (str(self.did))
            ws.Cells(i+2, 2).Value = (str(self.typename))
            ws.Cells(i+2, 3).Value = (str(self.devname))
            ws.Cells(i+2, 4).Value = (str(self.port))
            ws.Cells(i+2, 5).Value = (str(self.tid))
            ws.Cells(i+2, 6).Value = (str(self.rec_type))
            ws.Cells(i+2, 7).Value = (str(self.rec_fmt))
            ws.Cells(i+2, 8).Value = (str(self.name))
            ws.Cells(i+2, 9).Value = (str(self.unit))
            ws.Cells(i+2, 10).Value = (str(self.minval))
            ws.Cells(i+2, 11).Value = ( str(self.maxval))
            ws.Cells(i+2, 12).Value = (str(self.color))
            ws.Cells(i+2, 13).Value = (str(self.strate))
            ws.Cells(i+2, 14).Value = (str(self.adc_gain))
            ws.Cells(i+2, 15).Value = (str(self.adc_offset))
            ws.Cells(i+2, 16).Value = (str(self.montype))
            ws.Cells(i+2, 17).Value = (str(self.did))
            ws.Cells(i+2, 18).Value = (str(self.dt))
            ws.Cells(i+2, 19).Value = (str(self.tid))

        print("100% complete")
        wb.SaveAs('c:\\python\\' + filename +".xlsx")
        excel.Quit()

"""


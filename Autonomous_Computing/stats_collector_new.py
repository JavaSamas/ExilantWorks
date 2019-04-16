######################################
# Collect machine statistics .
# Needs admin access to run this script
# TODO: Add Apple file header
######################################
import psutil
import time
import socket
import datetime

DEBUG = True
REMOTE_HOST = '127.0.0.0'
REMOTE_PORT = 9000
THIS_HOST = socket.gethostname()

#----------------------------------
# Method to collect machine 
# health statistics
#----------------------------------
def collect_stats():
    stat_str = THIS_HOST
    now = datetime.datetime.now()
    stat_str += "|" + now.strftime("%Y-%m-%d %H:%M:%S")
    #Get CPU statistics
    cpu_utilization_percent = psutil.cpu_percent(interval=2)
    cpu_times = psutil.cpu_times()
    cpu_iowait_millis = -1.00 
    cpu_context_switches = psutil.cpu_stats()[0]
    if (len(cpu_times) > 4) :  cpu_iowait_millis = cpu_times[4]
    if(DEBUG) :
        print("CPU Stats-------")
        print("Utilization=" + str(cpu_utilization_percent))
        print("IOWait=" + str(cpu_iowait_millis))
        print("Context switches=" + str(cpu_context_switches))
    #construct CPU stat string
    stat_str += "|" + str(cpu_utilization_percent) + "|" + str(cpu_iowait_millis) + "|" +str(cpu_context_switches)

    #Get Memory statistics
    memory_available = psutil.virtual_memory()[1]
    if(DEBUG):
        print("Memory Stats---")
        print("Available memory=" + str(memory_available))
    #construct Memory stat string
    stat_str +=  "|" + str(memory_available)

    #Get disk statistics
    disk_read_count=psutil.disk_io_counters()[0]
    disk_write_count=psutil.disk_io_counters()[1]
    disk_read_time=psutil.disk_io_counters()[4]
    disk_write_time=psutil.disk_io_counters()[5]
    if(DEBUG):
        print("Disk Stats---")
        print("Read count=" + str(disk_read_count))
        print("Write count="+str(disk_write_count))
        print("Read time="+str(disk_read_time))
        print("Write time=" + str(disk_write_time))

    #Construct disk stat string
    stat_str += "|" + str(disk_read_count) + "|" + str(disk_write_count)+ "|" + str(disk_read_time)+ "|" + str(disk_write_time)

    #Get network statistics
    net_connections = len(psutil.net_connections())
    net_errin = psutil.net_io_counters()[4]
    if(DEBUG):
        print("Network Stats---")
        print("Open connections=" + str(net_connections))
        print("Receive errors= "+str(net_errin))

    #Construct network stat string
    stat_str += "|" + str(net_connections) + "|" + str(net_errin)

    return stat_str
#------------------------------------
# Method to send health statistics
# to remote server
#------------------------------------
def send_stats(stat):
    stat += str("\r\n")
    try:
        ##TODO::make calls to Geneva Q
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.settimeout(5.0) #5 second timeout
            s.connect((REMOTE_HOST,REMOTE_PORT))
            s.sendall(stat)
            s.close()
    except:
        print("Failed to connect to remote host")

#Collect stats and send to remote
while True:
    try:
        stat_str_collected = collect_stats()
        #TODO::send_stats(stat_str_collected)
        if(DEBUG):
            print("Collected Stats-----")
            print(stat_str_collected)
        time.sleep(1)
    except:
        time.sleep(1) #do nothing

import serial
import threading
import msvcrt
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime


SERIAL_PORT = 'COM4'
SERIAL_RATE = 115200
ser = serial.Serial(SERIAL_PORT, SERIAL_RATE,
                    stopbits=serial.STOPBITS_ONE, timeout=None)

dataPath = f"data\\{datetime.now().strftime('%Y-%m-%d')}.csv"

# Create figure for plotting

plt.style.use('seaborn-deep')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # figure(figsize=[10, 6])


timeData = []
tempData = []
humData = []
pressData = []

packet = []
countery = 0
counterx = 0
pressure = 0
temperature = 0
humidity = 0
lastPacketIndex = 0
lostPacketCount = 0
packetLength = 18
packetIndex = 0
bytesToRead = 0
eco2 = 0
tvoc = 0


def receive_packet():
    # while True:

    global packet
    global humidity
    global pressure
    global temperature
    global lostPacketCount
    global packetIndex
    global lastPacketIndex
    global eco2
    global tvoc

    bytesToRead = ser.inWaiting()
    while(bytesToRead < 26):
        bytesToRead = ser.inWaiting()

    packet = ser.read(bytesToRead)

    start = packet.find(b'\xaa')

    if packet[start+25] == 0xff:
        temperature = (int.from_bytes(
            packet[start+1:start+5], byteorder='big')*0.01)
        humidity = (int.from_bytes(
            packet[start+5:start+9], byteorder='big')*(1.0 / 1024.0))
        pressure = (int.from_bytes(
            packet[start+9:start+13], byteorder='big')*0.0001)
        eco2 = (int.from_bytes(
            packet[start+13:start+17], byteorder='big'))
        tvoc = (int.from_bytes(
            packet[start+17:start+21], byteorder='big'))
        packetIndex = (int.from_bytes(
            packet[start+21:start+25], byteorder='big'))

    if lastPacketIndex != packetIndex:
        lostPacketCount = lostPacketCount+(packetIndex-lastPacketIndex)-1
        print(
            f"packet index: {packetIndex} lost packets: {lostPacketCount} total packet loss: {round((lostPacketCount/packetIndex)*100)}%")
        if(lostPacketCount < 0):
            lostPacketCount = 0

    lastPacketIndex = packetIndex

    # print(packetIndex)

    # time.sleep(0.01)
    # time.sleep(1)


# Todo data collection function to write date to CSV


def animate_real_time(i, xs, ys, zs, ds):
    global countery
    global counterx
    global packet
    global humidity
    global pressure
    global temperature
    countery += 2
    counterx += 1

    #! Add x and y to lists
    # datetime.now().strftime('%M:%S.%f')[:-4]
    xs.append(datetime.now().strftime('%M:%S.%f')[:-4])
    ys.append(temperature)
    zs.append(humidity)
    ds.append(pressure)

    #! Limit x and y lists to 30 items

    xs = xs[-30:]
    ys = ys[-30:]
    zs = zs[-30:]
    ds = ds[-30:]
    #! delete after 40 elements to save memory

    if len(xs) == 20:
        del xs[0:]
        del ys[0:]
        del zs[0:]
        del ds[0:]

    #! Draw and format x/y lists
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.plot(xs, ys, '-ro', alpha=0.5)
    ax1.title.set_text('Temperature')
    ax1.set_xticks([])
    ax2.plot(xs, zs, '-bo', alpha=0.5)
    ax2.set_xticks([])

    ax2.title.set_text('Humidity')

    ax3.plot(xs, ds, '-go', alpha=0.5)
    ax3.title.set_text('Pressure')

    ax1.patch.set_facecolor('white')
    plt.xticks(rotation=45, ha='right')
    # plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.95)


try:
    #! Real time block
    # threadas = threading.Thread(
    #     target=receive_packet, args=[], daemon="true")
    # ani = animation.FuncAnimation(
    #     fig, animate_real_time, fargs=(timeData, tempData, humData, pressData), interval=100)
    # threadas.start()

    # plt.show()
    # ser.flushInput()
    # ser.close()
    #! end
    flag = True
    #! data log block
    while flag == True:
        receive_packet()
        print(
            f"{datetime.now().strftime('%H:%M:%S.%f')[:-4]} Temp: {round(temperature,2)} Humid: {round(humidity,2)} Press: {round(pressure,4)} Eco2: {eco2} Tvoc: {tvoc}\n")
        with open(dataPath, 'a') as file_obj:
            file_obj.write(
                f"{datetime.now().strftime('%H:%M:%S.%f')[:-4]},{round(temperature,2)},{round(humidity,2)},{round(pressure,4)},{eco2},{tvoc},\n")  # todo fix excel datetime
    file_obj.close()

except KeyboardInterrupt:
    ser.close()
    flag = False
    pass

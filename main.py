import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import math
import wave
import numpy as np

#midi化関数
def data2midi(F, fs, N):
    sec = N / fs
    loopcount, beforenote, maxvolume = 0, 0, 0
    for i in range(1, int(N/2), 1):
        if abs(F.imag[i]/N*2) > 8: #音量の閾値
            if 66 < i/sec and i/sec < 11175	: #midiの範囲に収める
                #ノート番号計算
                log = math.log10(i/sec/440) #i/secが周波数
                midinote = round(69 + log/0.025085832971998432934478241227, 1)

                #音量計算
                volume = (abs(F.imag[i]/N*2) ** (1.8/3))*1.25
                #音量調整
                if midinote < 56:
                    volume = volume * 0.8
                    if midinote < 48:
                        volume = volume * 0.7
                        if midinote < 41:
                            volume = volume * 0.6

                if midinote > 107:
                    volume = volume * 0.8
                    if midinote > 112:
                        volume = volume * 0.7
                        if midinote > 118:
                            volume = volume * 0.6

                if volume > 127: volume = 127

                loopcount += 1
                if not beforenote == int(round(midinote, 0)):
                    syosu, seisu = math.modf(beforenote) #整数と小数部分の分離
                    syosu = round(syosu, 1)

                    if syosu == 0.0:
                        track.append(Message('note_on', note=beforenote, velocity=int(round(maxvolume*1, 0)), time=00))
                    elif syosu == 0.9 or syosu == 0.8 or syosu == 0.1 or syosu == 0.2:
                        track.append(Message('note_on', note=beforenote, velocity=int(round(maxvolume*0.9, 0)), time=00))
                    elif syosu == 0.7 or syosu == 0.3:
                        track.append(Message('note_on', note=beforenote, velocity=int(round(maxvolume*0.8, 0)), time=00))
                    elif syosu == 0.6 or syosu == 0.4:
                        track.append(Message('note_on', note=beforenote, velocity=int(round(maxvolume*0.7, 0)), time=00))
                    elif syosu == 0.5:
                        track.append(Message('note_on', note=beforenote, velocity=int(round(maxvolume*0.6, 0)), time=00))

                    beforenote, maxvolume = int(round(midinote, 0)), volume
                    loopcount = 0
                else:
                    if volume > maxvolume: maxvolume = volume

    for j in range(35, 126):
        if j == 35:
            soundtime = 240*sec
            track.append(Message('note_off', note=j, time=int(round(soundtime, 0))))
        else:
            track.append(Message('note_off', note=j, time=0))


# Wave読み込み
def read_wav(file_path):
    wf = wave.open(file_path, "rb")
    buf = wf.readframes(-1) # 全部読み込む

    # 16bitごとに10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    else:
        data = 0

    # ステレオの場合，チャンネルを分離
    if wf.getnchannels()==2:
        data_l = data[::2]
        data_r = data[1::2]
    else:
        data_l = data
        data_r = data
    wf.close()
    return data_l,data_r


# wavファイルの情報を取得
def info_wav(file_path):
    ret = {}
    wf = wave.open(file_path, "rb")
    ret["ch"] = wf.getnchannels()
    ret["byte"] = wf.getsampwidth()
    ret["fs"] = wf.getframerate()
    ret["N"] = wf.getnframes()
    ret["sec"] = ret["N"] / ret["fs"]
    wf.close()
    return ret


# データ分割
def audio_split(data_l, wi, win_size):
    ret_ls = []
    win = np.hanning(win_size)
    for i in range(0, wi["N"] ,int(win_size)):
        endi = i + win_size
        if endi < wi["N"]:
            ret_ls.append(data_l[i:endi] * win)
        else:
            win = np.hanning(len(data_l[i:-1]))
            ret_ls.append(data_l[i:-1] * win)
    return ret_ls


if __name__ == '__main__':
    
    # midi定義
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
 
    # Wav読み込み
    data_l,data_r = read_wav("test.wav")
    del data_r

    # Wavの情報取得
    wi = info_wav("test.wav")
        
    # ウィンドウサイズ
    win_size = 1024 * 8
    
    #テンポ(実データ速度に近似)
    audiospeed = 16 / 8
    miditempo = (480/(wi["fs"]/win_size))/audiospeed
    miditempo = round(480 * (round(miditempo, 0) / miditempo) - audiospeed, 2)
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(miditempo)))
    
    # データ分割
    data_ls = audio_split(data_l, wi, win_size)
    del data_l

    # FFT&midi化
    for i in range(0, int(len(data_ls))):
        f_dl = np.fft.fft(data_ls[i])
        data2midi(f_dl, wi["fs"], len(f_dl.imag))

    out_file = "test_wav_midi.mid"
    mid.save(out_file)

import pandas as pd
import scipy.io.wavfile
import librosa
import soundfile as sf

import os
import mido
import shutil
# from midi2audio import FluidSynth

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

import copy

class FileBrowser:
    global player
    def __init__(self, parent):
        self.parent = parent
        # Define the instruments and volumes
        self.instruments = ['piano', 'bass', 'synth', 'drums']
        self.volumes = []  # Adjust these values based on the sliders
        self.path = None
        self.isSoundEdited = False
        self.frequencyRanges = {"animal sounds": [[650, 0, 950, 3000],[950, 650, 1900, 8100]],
                                "ECG abn": [[120, 1101, 51], [200, 1200, 60]],
                                "uniform": [[0, 13, 26, 39, 52, 65, 78, 91, 104, 117], [13, 26, 39, 52, 65, 78, 91, 104, 117, 130]]}
        # [0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130]

    def browse_file(self, mode):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        if mode == "uniform":
            self.path = QFileDialog.getOpenFileName(self.parent,"QFileDialog.getOpenFileName()", "","All Files (*);;CSV Files (*.csv);;DAT Files (*.dat);;XLSX Files (*.xlsx);;TXT Files (*.txt)", options=options)
        else:
            self.path = QFileDialog.getOpenFileName(self.parent,"QFileDialog.getOpenFileName()", "","Audio Files (*.mp3 *.wav *.ogg)", options=options)

        if self.path != ('', ''):
            if mode == "uniform":
                return self.read_file(self.path[0])
            else:
                # # Get the path of the original MIDI file
                # original_midi_path = os.path.splitext(self.path[0])[0] + '.mid'

                # # Define the path of the new MIDI file
                # new_midi_path = 'song_modified.mid'

                # # Copy the original MIDI file to the new MIDI file
                # shutil.copyfile(original_midi_path, new_midi_path)

                # samplingRate, audioData = scipy.io.wavfile.read(self.path[0])
                audioData, samplingRate = librosa.load(self.path[0])
                player = QMediaPlayer()
                player.setMedia(QMediaContent(QUrl.fromLocalFile(self.path[0])))

                return audioData, samplingRate, player
        else:
            return None, None

    def set_mode(self, mode):
        fMax = []
        fMin = []
        edit_labels = False
        # self.clearAll()
        # self.mode = mode
        isAudio = (mode == "musical instruments" or mode == "animal sounds" or mode == "ECG abn")
        # if isAudio:
        #     self.pushButton.setHidden(False)
        #     self.pushButton_2.setHidden(False)
        # else:
        #     self.pushButton.setHidden(True)
        #     self.pushButton_2.setHidden(True)
        modeNumOfSliders = {"musical instruments": 4, "animal sounds": 4, "ECG abn": 3, "uniform": 10} 
        numberOfSliders = modeNumOfSliders[mode]

        if mode in self.frequencyRanges.keys():
            fMin = self.frequencyRanges[mode][0]
            fMax = self.frequencyRanges[mode][1]
        # else:
        #     # self.slidersRange = []
        #     step = int(len(self.outputFrequencies)/self.numberOfSliders)
        #     for i in range(0, len(self.outputFrequencies) + 1, step):
        #         self.fMin.append(int(self.outputFrequencies[min(i, len(self.outputFrequencies) - 1)]))
        #         self.fMax.append(int(self.outputFrequencies[min(i, len(self.outputFrequencies) - 1)]))
        #     self.fMax.pop(0)
        if mode == "uniform":
            edit_labels = True
        #     for i in range(0, numberOfSliders):
        #         label = getattr(self, f"label_{i}_Hz")
        #         label.setText(f"{str(self.fMin[i])} Hz")
            # self.fMin.pop()

        return isAudio, numberOfSliders, fMin, fMax, edit_labels
    # def extract_duration(self, input_file, output_file, duration=5):
    #     # Load the audio file
    #     audio_data, sample_rate = librosa.load(input_file, sr=None)

    #     # Calculate the number of samples corresponding to the desired duration
    #     target_samples = int(duration * sample_rate)

    #     # Take only the first `target_samples` from the audio data
    #     extracted_audio = audio_data[:target_samples]

    #     # Write the extracted audio to a new file
    #     sf.write(output_file, extracted_audio, sample_rate)

    def setSlidersBands(self, mode):
        if self.mode in self.frequencyRanges.keys():
            self.fMin = self.frequencyRanges[self.mode][0]
            self.fMax = self.frequencyRanges[self.mode][1]
        else:
            # self.slidersRange = []
            step = int(len(self.outputFrequencies)/self.numberOfSliders)
            for i in range(0, len(self.outputFrequencies) + 1, step):
                self.fMin.append(int(self.outputFrequencies[min(i, len(self.outputFrequencies) - 1)]))
                self.fMax.append(int(self.outputFrequencies[min(i, len(self.outputFrequencies) - 1)]))
            self.fMax.pop(0)
        if self.mode == "uniform":
            for i in range(1, self.numberOfSliders + 1):
                label = getattr(self, f"label_{i}_Hz")
                label.setText(f"{str(self.fMin[i])} Hz")
            self.fMin.pop()

    def read_file(self, fileName):
        df = None  # Initialize df
        if fileName.endswith('.csv'):
            df = pd.read_csv(fileName)
        elif fileName.endswith('.xlsx'):
            df = pd.read_excel(fileName)
        elif fileName.endswith('.dat') or fileName.endswith('.txt'):
            df = pd.read_csv(fileName, sep='\t')
        time = df['Time'].values if df is not None else None
        amplitude = df['Amplitude'].values if df is not None else None
        return time, amplitude

    # def update_volume(self, index, value):
    #     if self.isSoundEdited == False:
    #         self.volumes = [0, 64, 0, 64, 64, 64]
    #         self.isSoundEdited = True
    #     if index > 1:
    #         volume_index = index + 1
    #     else:
    #         volume_index = index
    #     self.volumes[volume_index] = value

    #     # Get the path of the MIDI file with the same name
    #     midi_path = os.path.splitext(self.path[0])[0] + '.mid'

    #     # Load the MIDI file
    #     midi = mido.MidiFile(midi_path)
    #     for i, track in enumerate(midi.tracks):
    #         # Use the volume from the volumes list if it exists
    #         volume = self.volumes[i] if i < len(self.volumes) else 0  # Default volume for any additional tracks

    #         for msg in track:
    #             if msg.type == 'note_on' or msg.type == 'note_off':
    #                 # Adjust the velocity of the notes
    #                 msg.velocity = int(msg.velocity * volume / 127)
    #             elif msg.type == 'control_change' and msg.control == 7:
    #                 # Adjust the value of volume-related control change events
    #                 msg.value = int(msg.value * volume / 127)


        # # Save the modified MIDI file
        # midi.save(r'D:\3-1\DSP\task3_1\task3_1\song_modified.mid')

        # # Specify the full path to the output file
        # wav_path = os.path.abspath(r'D:\3-1\DSP\task3_1\task3_1\song_modified.wav')
        # # Check if the file exists
        # if os.path.exists(wav_path):    
        #     # Delete the file
        #     os.remove(wav_path)

        # # Create a FluidSynth instance
        # fs = FluidSynth('C:\ProgramData\soundfonts\default.sf2')

        # # Convert the MIDI file to a WAV file    
        # fs.midi_to_audio(r'D:\3-1\DSP\task3_1\task3_1\song_modified.mid', wav_path)

        # # Load the audio data
        # audioData, samplingRate = librosa.load(wav_path)

        # # Create a media player
        # player = QMediaPlayer()
        # player.setMedia(QMediaContent(QUrl.fromLocalFile(wav_path)))

        # return audioData, samplingRate, player


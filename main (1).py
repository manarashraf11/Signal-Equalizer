import os
import numpy as np
import pandas as pd
# import pyqtgraph as pg
# import librosa
import soundfile as sf
import scipy
from scipy import signal
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.offline as pyo
# import sounddevice as sd
import copy
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox, QApplication, QVBoxLayout, QWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon
import sys
import task3ui
from classes_1 import FileBrowser

class MainApp(QtWidgets.QMainWindow, task3ui.Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.inputViewBox = self.PlotWidget_inputSignal.getViewBox()
        self.outputViewBox = self.PlotWidget_outputSignal.getViewBox()
        self.inputViewBox.setXLink(self.outputViewBox)
        self.inputViewBox.setYLink(self.outputViewBox)
        self.PlotWidget_fourier.setLabel('left', 'Magnitude')
        self.PlotWidget_fourier.setLabel('bottom', 'Frequency')

        # self.fig_input = Figure()
        # self.canvas_input = FigureCanvas(self.fig_input)
        # layout_input = self.groupBox_4.layout()
        # layout_input.addWidget(self.canvas_input)

        # self.fig_output = Figure()
        # self.canvas_output = FigureCanvas(self.fig_output)
        # layout_output = self.groupBox_8.layout()
        # layout_output.addWidget(self.canvas_output)

        def create_figure_and_canvas(group_box):
            fig = Figure()
            canvas = FigureCanvas(fig)
            layout = group_box.layout()
            layout.addWidget(canvas)
            return fig, canvas

        self.fig_input, self.canvas_input = create_figure_and_canvas(self.groupBox_4)
        self.fig_output, self.canvas_output = create_figure_and_canvas(self.groupBox_8)

        self.checkBox_showSpectrogram.setChecked(True)
        self.spinBox_standardDeviation.setDisabled(True)
        self.spinBox_speed.setRange(1, 7)
        self.spinBox_speed.setValue(4)
        self.spinBox_speed.valueChanged.connect(lambda: self.setSpeed(self.spinBox_speed.value()))

        self.fileBrowser = FileBrowser(self)
        self.mode = "uniform"
        self.inputSignal = []
        self.audio_input = np.array([]) 
        self.audio_output = np.array([]) 
        self.audioTimeValues = []
        self.audioFs = 0
        self.outputSignal = []
        self.mediaPlayer_input = QMediaPlayer()
        self.mediaPlayer_input.setVolume(50)
        self.mediaPlayer_output = QMediaPlayer()
        self.mediaPlayer_output.setVolume(50)
        self.window_type = 'Rectangular window'
        self.comboBox_smoothingWindows.setCurrentText(self.window_type)
        self.endOfViewRange = 0
        self.playing = False
        self.outputFrequencies = []
        self.outputMagnitudes = []
        self.outputPhase = []
        self.numberOfSliders = 10
        self.isAudio = False
        self.std = None
        self.numModifiedSd = 0
        self.axisStep = 0
        self.graphsTimer = QtCore.QTimer()
        self.graphsTimer.timeout.connect(self.animateAxis)
        self.graphsTimer.setInterval(10)
        self.inputSoundOn = True
        self.pushButton_2.setHidden(True)
        self.pushButton_stop.setEnabled(False)
        self.pushButton_reset.setEnabled(False)
        
        self.label_14.setText("10 dB")
        self.label_17.setText("-10 dB")
        self.label_15.setText("20 dB")
        self.label_18.setText("-20 dB")

        self.actionOpen.triggered.connect(self.uploadAndPlotSignal)
        self.actionClose.triggered.connect(self.closeApp)

        mode_actions = {
            self.actionUniform: "uniform",
            self.actionMusical: "musical instruments",
            self.actionAnimal_Sounds: "animal sounds",
            self.actionECG_Abnormalities: "ECG abn",
        }

        for action, mode in mode_actions.items():
            action.triggered.connect(lambda _, m=mode: self.setMode(m))

        self.pushButton_playPause.clicked.connect(self.togglePlayPause)
        self.pushButton_zoomIn.clicked.connect(lambda: self.zoom(0.5))
        self.pushButton_zoomOut.clicked.connect(lambda: self.zoom(2))
        self.pushButton_reset.clicked.connect(lambda: self.stopAndReset(True))
        self.pushButton_stop.clicked.connect(lambda: self.stopAndReset(False))
        self.pushButton_applySmoothing.clicked.connect(self.setWindowParameters)
        self.pushButton.clicked.connect(self.toggleMuteUnmute)
        self.pushButton_2.clicked.connect(self.toggleMuteUnmute)
        self.pushButton.setHidden(True)

        self.muteIcon = QIcon("icons/mute2.png")
        self.unmuteIcon = QIcon("icons/unmute2.png")
        self.pushButton_2.setIcon(self.muteIcon)

        self.checkBox_showSpectrogram.stateChanged.connect(self.showAndHideSpectrogram)
        self.comboBox_smoothingWindows.activated.connect(self.setSmoothingWindow)

        for i in range(1, self.numberOfSliders + 1):
            slider = getattr(self, f"verticalSlider_{i}")
            slider.setRange(0, 10)
            slider.setSingleStep(1)
            slider.setValue(5)
            slider.valueChanged.connect(lambda value, index=i: self.generateWindow(index, value))

        self.frequencyRanges = {"animal sounds": [[650, 0, 950, 3000],[950, 650, 1900, 8100]],
                                "ECG abn": [[120, 1101, 51], [200, 1200, 60]]}
        self.outputChanged = False
        self.fMax = []
        self.fMin = []

    def closeApp(self):
        QApplication.instance().quit()

    def setMode(self, mode):
        self.clearAll()
        self.mode = mode
        self.isAudio, self.numberOfSliders, self.fMin, self.fMax, edit_labels = FileBrowser.set_mode(self.fileBrowser, mode)
        if edit_labels == True:
            for i in range(0, self.numberOfSliders):
                label = getattr(self, f"label_{i + 1}_Hz")
                label.setText(f"{str(self.fMax[i])} Hz")

    def uploadAndPlotSignal(self):
        self.clearAll()
        file_data = self.fileBrowser.browse_file(self.mode)
        if self.isAudio:
            self.audio_input, self.audioFs, self.mediaPlayer_input = file_data
            # Copy the media source
            self.mediaPlayer_output.setMedia(self.mediaPlayer_input.media())
            # Copy the position
            self.mediaPlayer_output.setPosition(self.mediaPlayer_input.position())
            # Copy the volume
            self.mediaPlayer_output.setVolume(self.mediaPlayer_input.volume())
            if self.audio_input.ndim == 2: 
                self.audio_input = self.audio_input[:, 0]
            self.audio_output = copy.copy(self.audio_input)
        else:
            time, amplitude = file_data
            self.inputSignal, self.outputSignal = [amplitude, time], [amplitude, time] 

        # plot the input and output
        self.plotSignal_timeDomain(self.audio_input, self.audioFs, self.inputSignal, self.PlotWidget_inputSignal)

        self.plotSignal_timeDomain(self.audio_output, self.audioFs, self.outputSignal, self.PlotWidget_outputSignal)
        
        self.inputSpec = self.plotSpectrogram(self.fig_input, self.canvas_input, self.audio_input, self.audioFs, self.inputSignal)
        self.outputSpec = self.plotSpectrogram(self.fig_output, self.canvas_output, self.audio_output, self.audioFs, self.outputSignal)

        # compute fourier transform of the input
        self.outputMagnitudes, self.outputPhase, self.outputFrequencies = self.computeFourierTransform()
        self.newMagnitudes = copy.copy(self.outputMagnitudes)

        self.plotFrequencySpectrum()
        # self.setSlidersBands()

    def plotSignal_timeDomain(self, audio, audioSamplingRate, signal, widget):
        if self.isAudio:
                peak_value = np.amax(audio)
                normalized_data = audio / peak_value
                amplitude_values = normalized_data
                length = audio.shape[0] / audioSamplingRate
                time_values = list(np.linspace(0, length, audio.shape[0]))
                self.audioTimeValues = copy.copy(time_values)
        else:
            time_values = signal[1]
            amplitude_values = signal[0]
        # plot input signal
        widget.clear()
        widget.plot(time_values, amplitude_values, pen="b")
        widget.setLabel('left', 'Amplitude')
        widget.setLabel('bottom', 'Time')
        min_y, max_y = self.get_min_max_for_widget(widget, "y")
        self.inputViewBox.setYRange(min_y, max_y)
        self.inputViewBox.setLimits(xMin=time_values[0] - 0.1, xMax=time_values[-1] + 0.1, yMin=min_y - 0.1, yMax=max_y + 0.1)
        self.outputViewBox.setYRange(min_y, max_y)
        self.outputViewBox.setLimits(xMin=time_values[0] - 0.1, xMax=time_values[-1] + 0.1, yMin=min_y - 0.1, yMax=max_y + 0.1)

    def plotSpectrogram(self, fig, canvas, audio, audioSamplingRate, signal):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.clear()  # Clear the previous spectrogram
        if self.isAudio:
            ax.specgram(audio, Fs=audioSamplingRate, cmap='viridis')
        else:
            ax.specgram(signal[0], Fs=1/(signal[1][1] - signal[1][0]), cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel("Frequency")
        canvas.draw()
        # self.outputSpec = ax
        return ax

    def plotFrequencySpectrum(self):
        self.PlotWidget_fourier.clear()
        self.PlotWidget_fourier.plot(self.outputFrequencies, self.newMagnitudes, pen="c")

    def computeFourierTransform(self):
        if self.isAudio:
            signal = self.audio_input
            sampleRate = self.audioFs
        else:
            signal = self.inputSignal[0]
            sampleRate = 1 / (self.inputSignal[1][1] - self.inputSignal[1][0])

        # Compute the Real FFT
        complex_fft = scipy.fft.rfft(signal)

        # Calculate the magnitude and phase
        magnitude = np.abs(complex_fft)
        phase = np.angle(complex_fft)

        # Calculate the self.outputFrequencies values corresponding to the FFT result
        self.outputFrequencies = scipy.fft.rfftfreq(len(signal), 1 / sampleRate)

        return magnitude, phase, self.outputFrequencies

    def invFourierTransform(self, magnitude, phase):
        # Reconstruct the complex Fourier transform from magnitude, phase, and self.outputFrequencies
        complex_fft = magnitude * np.exp(1j * phase)
        
        # Perform the inverse Fourier transform
        reconstructedSignal = np.fft.irfft(complex_fft)
        
        return reconstructedSignal

    def setSmoothingWindow(self):
        # get the selected window
        self.window_type = self.comboBox_smoothingWindows.currentText()
        # En/disable std based on the selected window type
        if self.window_type in ['Hamming window', 'Hann window', 'Rectangular window']:
            self.spinBox_standardDeviation.setDisabled(True)
        else:
            self.spinBox_standardDeviation.setDisabled(False)

    def setWindowParameters(self):
        self.setSmoothingWindow()
        if self.spinBox_standardDeviation.isEnabled():
            self.std = self.spinBox_standardDeviation.value()
        else:
            self.std = None
        # self.generateWindow()

    def getMappedSliderValue(self, slider_value):
        input_min = self.verticalSlider_1.minimum()
        input_max = self.verticalSlider_1.maximum()
        output_min = -40
        output_max = 40

        mapped_value = (slider_value - input_min) * (output_max - output_min) / (input_max - input_min) + output_min
        return mapped_value

    def generateWindow(self, sliderNumber, Value):
        self.outputChanged = True
        dbValue = self.getMappedSliderValue(Value)
        if self.window_type == "Gaussian window" and self.std == None:
            # Create a QMessageBox and display the critical error dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please Finish Setting Up the Smoothing Window")
            msg_box.exec_()
        else:
            # # define target frequency band based on the slider number
            # targetBand = list(np.where((self.outputFrequencies >= self.slidersRange[sliderNumber - 1]) & (self.outputFrequencies <= (self.slidersRange[sliderNumber - 1] + self.slidersRange[1])))[0])
            targetBand = []
            if self.mode == "musical instruments":
                self.audio_output, self.audioFs, self.mediaPlayer_output = self.fileBrowser.update_volume(sliderNumber, (Value / 10) * 127)
                self.plotSignal_timeDomain(self.audio_output, self.audioFs, self.outputSignal, self.PlotWidget_outputSignal)
                self.outputSpec = self.plotSpectrogram(self.fig_output, self.canvas_output, self.audio_output, self.audioFs, self.outputSignal)
            else:
                targetBand = list(np.where((self.outputFrequencies >= self.fMin[(sliderNumber - 1)])  & (self.outputFrequencies <= self.fMax[(sliderNumber - 1)]))[0])
            window_types = {
            'Rectangular window': signal.windows.boxcar,
            'Hamming window': signal.windows.hamming,
            'Hann window': signal.windows.hann,
            }

            # generate selected window
            selected_window_function = window_types[self.window_type]
            if self.window_type in window_types:
                gain = 10**(dbValue/20) * selected_window_function(len(targetBand))
            else:
                gain = 10**(dbValue/20) * signal.windows.gaussian(len(targetBand), self.std)
            # plot all after applying smoothing window
            self.applySmoothingWindow(gain, targetBand)

    def applySmoothingWindow(self, gainList, targetBand):
        if self.playing:
            self.togglePlayPause()
        # get the band frequency values
        frequency = self.outputFrequencies[targetBand]
        # compute the corresponding magnitude values
        for i, gain in zip(targetBand, gainList):
            self.newMagnitudes[i] = self.outputMagnitudes[i] * gain

        # plot window and the new frequency spectrum
        self.plotFrequencySpectrum()
        self.PlotWidget_fourier.plot(frequency, gainList * max(self.newMagnitudes), pen="m")

        if self.isAudio:
            self.audio_output = self.invFourierTransform(self.newMagnitudes, self.outputPhase)
        else:
            self.outputSignal[0] = self.invFourierTransform(self.newMagnitudes, self.outputPhase)
        # plot the modified output
        self.plotSignal_timeDomain(self.audio_output, self.audioFs, self.outputSignal, self.PlotWidget_outputSignal)
        self.outputSpec = self.plotSpectrogram(self.fig_output, self.canvas_output, self.audio_output, self.audioFs, self.outputSignal)

    def togglePlayPause(self):
        if self.playing:
            if self.isAudio:
                self.mediaPlayer_input.pause()
                self.mediaPlayer_output.pause()
            self.graphsTimer.stop()
            self.playing = False
        else:
            self.setSpeed(self.spinBox_speed.value())
            if self.isAudio:
                # check if output is modified
                if self.outputChanged:
                    print(f"new audio{self.numModifiedSd}")
                    self.outputChanged = False
                    # Convert the signal to 16-bit PCM WAV format
                    # newSignalWav = np.int16(self.audio_output * 32767)
                    # scipy.io.wavfile.write(f'modified_audio{self.numModifiedSd}.wav', self.audioFs, newSignalWav)
                    # Using librosa
                    # librosa.output.write_wav(f'modified_audio{self.numModifiedSd}.wav', self.audio_output, self.audioFs)
                    audioPaths = {
                        "animal sounds" : f'modified_audio{self.numModifiedSd}.wav',
                        "musical instruments" : 'song_modified.wav'
                    }
                    # Check if the file exists
                    for i in range(self.numModifiedSd):
                        if os.path.exists(f'modified_audio{i}.wav'):
                            # Delete the existing file
                            try:
                                os.remove(f'modified_audio{i}.wav')
                                print(f"Deleted existing file: {f'modified_audio{i}.wav'}")
                            except Exception as e:
                                print(f"Error deleting file: {e}")

                        # Write the signal to a WAV file
                        sf.write(f'modified_audio{self.numModifiedSd}.wav', self.audio_output, self.audioFs)
                    
                    self.numModifiedSd += 1

                    # Set the media content to the WAV file
                    self.mediaPlayer_output.setMedia(QMediaContent(QUrl.fromLocalFile(audioPaths[self.mode])))
                    print(self.mediaPlayer_output.media)
                self.mediaPlayer_input.play()
                self.mediaPlayer_output.play()
                # else:
                    # # Copy the media source
                    # self.mediaPlayer_output.setMedia(self.mediaPlayer_input.media())
                    # # Copy the position
                    # self.mediaPlayer_output.setPosition(self.mediaPlayer_input.position())
                    # # Copy the volume
                    # self.mediaPlayer_output.setVolume(self.mediaPlayer_input.volume())
                # self.mediaPlayer_input.play()
                # self.mediaPlayer_output.play()
            self.graphsTimer.start()
            self.playing = True

    def animateAxis(self):
        self.pushButton_stop.setEnabled(True)
        self.axisStep += 0.01
        if self.isAudio:
            maxTimeValue = max(self.audioTimeValues)
        else:
            maxTimeValue = self.inputSignal[1][-1]
        if self.axisStep < maxTimeValue:
            self.inputViewBox.setXRange(self.axisStep, 0.8 + self.axisStep)
        else:
            self.pushButton_stop.setEnabled(False)
            self.pushButton_playPause.setEnabled(False)
            self.pushButton_reset.setEnabled(True)
            self.isPlaying = False
            self.mediaPlayer_input.stop()
            self.mediaPlayer_output.stop()
            self.graphsTimer.stop()

    def setSpeed(self, speed):
        # Define the mapping using a dictionary
        mapping = {7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7}

        # Use the dictionary to get the mapped speed value
        mapped_speed = mapping[speed]

        self.graphsTimer.setInterval((mapped_speed + 1) * 2)

        # Set the playback rate of the QMediaPlayer
        playback_rate = speed / 4
        self.mediaPlayer_input.setPlaybackRate(playback_rate)
        self.mediaPlayer_output.setPlaybackRate(playback_rate)

    def toggleMuteUnmute(self):
        if self.isAudio:
            # Toggle between input and output sound
            self.inputSoundOn = not self.inputSoundOn

            # Set mute states and update icons based on inputSoundOn
            self.mediaPlayer_input.setMuted(not self.inputSoundOn)
            self.mediaPlayer_output.setMuted(self.inputSoundOn)

            # Update button icons based on inputSoundOn
            self.pushButton.setIcon(self.unmuteIcon if self.inputSoundOn else self.muteIcon)
            self.pushButton_2.setIcon(self.muteIcon if self.inputSoundOn else self.unmuteIcon)

    def stopAndReset(self, reset):
        # stop playing and return to the beginning
        self.isPlaying = False
        self.axisStep = 0
        self.mediaPlayer_input.stop()
        self.mediaPlayer_output.stop()
        # self.mediaPlayer_output.setVolume(50)
        # self.mediaPlayer_input.setVolume(50)
        self.graphsTimer.stop()

        self.pushButton_stop.setEnabled(False)
        self.pushButton_reset.setEnabled(False)
        self.pushButton_playPause.setEnabled(True)

        self.endOfViewRange = 0
        self.inputViewBox.autoRange()
        # reset all view settings
        if reset:
            self.spinBox_speed.setValue(4)
            self.graphsTimer.setInterval(8)
            # self.mediaPlayer_input.setPlaybackRate(1/3)
            # self.mediaPlayer_output.setPlaybackRate(1/3)

    def zoom(self, factor):
        self.inputViewBox.scaleBy((factor, factor))

    def showAndHideSpectrogram(self, state):
        if state == QtCore.Qt.Checked:
            self.inputSpec.set_visible(True)
            self.outputSpec.set_visible(True)
        else:
            self.inputSpec.set_visible(False)
            self.outputSpec.set_visible(False)

        # Redraw the canvas to apply the changes
        self.canvas_input.draw()
        self.canvas_output.draw()

    def get_min_max_for_widget(self, widget, data_type):
        min_value = float('inf')
        max_value = float('-inf')

        for plot in widget.getPlotItem().listDataItems():
            data = getattr(plot, f"{data_type}Data")
            if data is not None:
                min_value = min(min_value, min(data))
                max_value = max(max_value, max(data))

        return min_value, max_value

    def clearAll(self):
        for i in range(1, self.numberOfSliders + 1):
            slider = getattr(self, f"verticalSlider_{i}")
            slider.setValue(5)
        self.outputChanged = False
        # clear all plots
        self.PlotWidget_fourier.clear()
        self.PlotWidget_inputSignal.clear()
        self.PlotWidget_outputSignal.clear()
        # clear figures
        self.fig_input.clear()
        self.canvas_input.draw()
        self.fig_output.clear()
        self.canvas_output.draw()
        # clear all stored data 
        self.outputFrequencies = []
        self.inputSignal = []
        self.audio_input = pd.DataFrame()
        self.audio_output = pd.DataFrame()
        self.outputSignal = [[],[]]
        self.outputFrequencies = []
        self.outputMagnitudes = []
        self.outputPhase = []
        self.audioFs = 0
        self.endOfViewRange = 0
        self.isPlaying = False
        self.std = None
        self.numModifiedSd = 0
        self.stopAndReset(True)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
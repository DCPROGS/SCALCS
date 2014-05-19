#! /usr/bin/python

import os
import numpy as np

from PySide.QtCore import *
from PySide.QtGui import *

import dcpyps.dcio as dcio

__author__="remis"
__date__ ="$03-Jan-2010 15:26:00$"

class ConverterQT(QDialog):
    def __init__(self, parent=None):
        super(ConverterQT, self).__init__(parent)
        self.resize(500, 100)     # width, height in px

        self.data = None
        self.to_filename = ''
        self.path = "."
        
        grid = QGridLayout()
        grid.addWidget(QLabel("From file:"), 0, 0)
        grid.addWidget(QLabel("To file:"), 1, 0)
        self.txt_from_file = QLineEdit()
        grid.addWidget(self.txt_from_file, 0, 1)
        self.txt_to_file = QLineEdit()
        grid.addWidget(self.txt_to_file, 1, 1)
        but_from_file = QPushButton("Browse")
        grid.addWidget(but_from_file, 0, 2)
        but_to_file = QPushButton("Browse")
        grid.addWidget(but_to_file, 1, 2)
        but_convert = QPushButton("Convert")
        grid.addWidget(but_convert, 2, 1)
        but_quit = QPushButton("Quit")
        grid.addWidget(but_quit, 2, 2)
        self.setLayout(grid)

        self.connect(but_quit, SIGNAL("clicked()"), self, SLOT("close()"))
        self.connect(but_convert, SIGNAL("clicked()"), self.convert)
        self.connect(but_from_file, SIGNAL("clicked()"), self.load_from_file)
        self.connect(but_to_file, SIGNAL("clicked()"), self.browse_to_file)

        self.setWindowTitle("Convert to SCN file...")

    def load_from_file(self):
        ""
        filename, filt = QFileDialog.getOpenFileName(self,
                "Open a text file...", self.path,
                "TXT files (*.txt *.TXT);;All files (*.*)")
        if filename:
            self.path = os.path.split(str(filename))[0]
            self.txt_from_file.setText(filename)
            self.data = dcio.txt_load_one_col(filename)

    def browse_to_file(self):
        ""
        self.to_filename, filt = QFileDialog.getSaveFileName(self,
                "Save as SCN file...", ".scn",
                "SCN files (*.scn)")
        if self.to_filename:
            self.txt_to_file.setText(self.to_filename)

    def convert(self):
        "One type intervals saved as shut intervals in scan file. "

        nint = 2 * len(self.data) + 1
        intervals = np.ones((nint), dtype='float') * 100
        intervals[1::2] = self.data
        amplitudes = np.zeros((nint), dtype='int')
        amplitudes[0::2] += 1
        options = np.zeros((nint), dtype='b')
        dcio.scn_write(np.array(intervals), amplitudes, options,
            filename=self.to_filename, type='Converted 1 type of ints')

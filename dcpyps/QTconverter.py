#! /usr/bin/python

from PyQt4.QtCore import *
from PyQt4.QtGui import *

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
        filename = QFileDialog.getOpenFileName(self,
                "Open a text file...", self.path,
                "TXT files (*.txt *.TXT);;All files (*.*)")
        if filename:
            self.path = os.path.split(str(filename))[0]
            self.txt_from_file.setText(filename)
            self.data = dcio.txt_load_one_col(filename)

    def browse_to_file(self):
        ""
        self.to_filename = QFileDialog.getSaveFileName(self,
                "Save as SCN file...", ".scn",
                "SCN files (*.scn)")
        if self.to_filename:
            self.txt_to_file.setText(self.to_filename)

    def convert(self):
        ""
        dcio.scn_write_dummy(self.data, self.to_filename)
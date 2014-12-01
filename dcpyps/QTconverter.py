#! /usr/bin/python

import os
import struct
import time
import numpy as np
import pandas as pd

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
                "Open a file...", self.path,
                "Excel files (*.xls *.xlsx;;TXT files (*.txt *.TXT);;All files (*.*)")
        if filename:
            self.path, fname = os.path.split(str(filename))
            self.txt_from_file.setText(filename)
            type = fname.split('.')[-1]
            if type == 'xls' or type == 'xlsx':
                sheets = pd.ExcelFile(filename).sheet_names
                dialog = ExcelSheetDlg(sheets, self)
                if dialog.exec_():
                    sheet = dialog.returnSheet()
                self.amplitudes, self.intervals, self.flags = load_Clampfit_Excel_sheet(filename, sheet)
            else:    
                self.data = txt_load_one_col(filename)

            

    def browse_to_file(self):
        ""
        self.to_filename, filt = QFileDialog.getSaveFileName(self,
                "Save as SCN file...", ".scn",
                "SCN files (*.scn)")
        if self.to_filename:
            self.txt_to_file.setText(self.to_filename)

    def convert1(self):
        "One type intervals saved as shut intervals in scan file. "

        nint = 2 * len(self.data) + 1
        intervals = np.ones((nint), dtype='float') * 100
        intervals[1::2] = self.data
        amplitudes = np.zeros((nint), dtype='int')
        amplitudes[0::2] += 1
        options = np.zeros((nint), dtype='b')
        dcio.scn_write(np.array(intervals), amplitudes, options,
            filename=self.to_filename, type='Converted 1 type of ints')

    def convert(self):
        try:
            scn_write(np.array(self.intervals), self.amplitudes, self.flags,
                filename=self.to_filename, type='Converted 1 type of ints')
            msgBox = QMessageBox()
            msgBox.setText("Conversion finished!")
            msgBox.exec_()
        except ValueError:
            msgBox = QMessageBox()
            msgBox.setText("Conversion aborted :-) \n Something went wrong. \n Please, contact Remis.")
            msgBox.exec_()
        

class ExcelSheetDlg(QDialog):
    """
    Dialog to choose Excel sheet to load.
    """
    def __init__(self, sheetlist, parent=None):
        super(ExcelSheetDlg, self).__init__(parent)

        self.sheet = ''
        self.List = QListWidget()
        self.List.addItems(sheetlist)
    
        self.connect(self.List,
            SIGNAL("itemSelectionChanged()"),
            self.sheetSelected)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))

        layout1 = QHBoxLayout()
        layout1.addWidget(self.List)
        layout2 = QVBoxLayout()
        layout2.addLayout(layout1)
        layout2.addWidget(buttonBox)

        self.setLayout(layout2)
        self.resize(200, 300)
        self.setWindowTitle("Choose Excel sheet to load...")

    def sheetSelected(self):
        """
        Get selected sheet name.
        """
        self.sheet = self.List.currentItem().text()

    def returnSheet(self):
        """
        Return selected sheet name.
        """
        return self.sheet


def load_Clampfit_Excel_sheet(fname, sheet):
    """
    Convert Clampfit idealisation result saved as comma delimited .csv file.
    """
    
    record = pd.read_excel(fname, sheet, header=None)
    amplitudes = np.abs(record.iloc[1: , 6].values * record.iloc[1:, 2].values * 1000.0).astype(int)
    intervals = record.iloc[1: , 8].values
    flags = np.zeros((len(intervals)), dtype='b')
    return amplitudes, intervals, flags
    

def txt_load_one_col(filename):
    """"
    Read one column of data from a text file.
    """

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        if line != '\n':
            value = float(line.strip("\t\n"))    #divide lines into values at tabs
#            print 'value=', value
            data.append(value)

    print "number of original intervals =", len(lines)
    print "number of useful intervals =", len(data)
    return data

def scn_write(intervals, amplitudes, flags, calfac=1.0, ffilt=-1.0, rms=0.0,
        treso=0.0, tresg=0.0, Emem=0.0,
        filename='new_saved.SCN', type='simulated'):
    """
    Write binary SCAN (DCprogs: http://www.ucl.ac.uk/Pharmacology/dcpr95.html)
    format file.

    Parameters
    ----------
    """

    # Preapare header.
    iscanver, ioffset = -103, 154
    nint, avamp = len(intervals), np.average(amplitudes)
    title = '{0: <70}'.format(type) # char*70
    t = time.asctime()
    expdate = t[8:10] + '-' + t[4:7] + '-' + t[20:24] # '00-ooo-0000' char*11
    tapeID = '{0: <24}'.format(type) # char*24
    ipatch = 0 # integer32

    # Write header.
    fout = open(filename, 'wb')
    fout.write(struct.pack('iii', iscanver, ioffset, nint))
    fout.write(title + expdate + tapeID)
    fout.write(struct.pack('ififff', ipatch, Emem, 0, avamp, rms, ffilt))
    fout.write(struct.pack('fff', calfac, treso, tresg))

    # Write data block.
    fout.write(struct.pack('f'*nint, *intervals))
    fout.write(struct.pack('h'*nint, *amplitudes))
    fout.write(struct.pack('b'*nint, *flags))
    fout.close()
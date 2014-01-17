import numpy as np

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import cjumps
from dcpyps import scplotlib as scpl

import myqtcommon

class JumpMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(JumpMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Jumps')
        
        self.cjprofile = 'rcj'
        self.cjlen = 0.05
        self.cjstep = 5e-6
        self.cjwidth = 0.00001
        self.cjfunc = None
        self.cjargs = None
        
        plotJumpPopenAction = myqtcommon.createAction(self, 
            "&Concentration jump: Popen", self.onPlotCJumpPopen)
        plotJumpOccupanciesAction = myqtcommon.createAction(self, 
            "&Concentration jump: occupancies",
            self.onPlotCJumpOccupancies)
        plotJumpOnOffTauConc = myqtcommon.createAction(self, 
            "&Concentration jump: weighted on/off tau versus concentration",
            self.onPlotCJumpRiseVConc)
#        plotJump2PopenAction = self.createAction(self, 
#            "&Instant rise and exponential decay concentration jump: Popen", self.onPlotCJump2Popen)
            
        self.addActions([plotJumpPopenAction, plotJumpOccupanciesAction,
            plotJumpOnOffTauConc
#            , plotJump2PopenAction            
            ])
            
    def onPlotCJumpPopen(self):
        """
        Display concentration jump.
        """

        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            self.cjprofile = dialog1.return_par()
        dialog = CJumpParDlg(self, self.cjprofile, self.cjlen, self.cjstep, self.cjargs)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.cjfunc, self.cjargs = dialog.return_par()

        
        self.parent.txtPltBox.clear()
        str = ('===== CONCENTRATION JUMP =====\n' +
            'Concentration profile- green solid line.\n' +
            'Relaxation- blue solid line.\n' +
            '\nConcentration pulse profile:\n' +
            'Peak concentration = {0:.5g} mM\n'.format(self.cjargs[0] * 1000) +
            'Background concentration = {0:.5g} mM\n'.
            format(self.cjargs[1] * 1000))
        self.parent.txtPltBox.append(str)
            
        if self.cjprofile == 'rcj':
            str = ('10- 90% rise time = {0:.5g} microsec\n'
                .format(self.cjargs[4] * 1e+6) +
                '90- 10% decay time = {0:.5g} microsec\n'
                .format(self.cjargs[5] * 1e+6) +
                'Pulse width = {0:.5g} millisec\n'
                .format(self.cjargs[3] * 1000))
            self.parent.txtPltBox.append(str)
        elif self.cjprofile == 'instexp':
            self.parent.txtPltBox.append('Decay time constant = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square':
            self.parent.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square2':
            self.parent.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
            self.parent.txtPltBox.append('Interpulse interval = {0:.5g} millisec'
                .format(self.cjargs[4] * 1000))
        self.parent.txtPltBox.append("---\n")

        t, c, Popen, P  = cjumps.solve_jump(self.parent.mec, self.cjlen, self.cjstep,
            self.cjfunc, self.cjargs)
        maxP = max(Popen)
        maxC = max(c)
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(t * 1000, Popen,'b-', t * 1000, c1, 'g-')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        if self.cjprofile == 'instexp':
            self.parent.log.write('\n\nCalculated response to an instantan jump to {0:.5g} mM '.
                format(self.cjargs[0] * 1000) +
                'concentration with an exponential decay tau of {0:.5g} ms: '.
                format(self.cjargs[3] * 1000) +
                'maximal Popen- {0:.5g}'.format(maxP))
        elif ((self.cjprofile == 'rcj') or (self.cjprofile == 'square')):
            self.parent.log.write(cjumps.printout(self.parent.mec,
                self.cjargs[0], self.cjargs[3]))
        self.parent.present_plot = np.vstack((t, Popen, c, P))

    def onPlotCJumpOccupancies(self):
        """
        Display realistic concentration jump.
        """

        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            self.cjprofile = dialog1.return_par()
        dialog = CJumpParDlg(self, self.cjprofile, self.cjlen, self.cjstep, self.cjargs)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.cjfunc, self.cjargs = dialog.return_par()

        self.parent.txtPltBox.clear()
        str = ('===== REALISTIC CONCENTRATION JUMP =====\n' +
            'Concentration profile- black solid line on top.\n' +
            'Popen relaxation- black solid line.\n' +
            'Occupancies of open states- red dashed lines.\n' +
            'Occupancies of shortlived shut states- green dashed lines.\n' +
            'Occupancies of longlived shut states- blue dashed lines.\n' +

            '\nConcentration pulse profile:\n' +
            'Peak concentration = {0:.5g} mM\n'
            .format(self.cjargs[0] * 1000) +
            'Background concentration = {0:.5g} mM\n'
            .format(self.cjargs[1] * 1000))
        self.parent.txtPltBox.append(str)
            
        if self.cjprofile == 'rcj':
            str = ('10- 90% rise time = {0:.5g} microsec\n'
                .format(self.cjargs[4] * 1e+6) +
                '90- 10% decay time = {0:.5g} microsec\n'
                .format(self.cjargs[5] * 1e+6) +
                'Pulse width = {0:.5g} millisec\n' 
                .format(self.cjargs[3] * 1000))
            self.parent.txtPltBox.append(str)
        elif self.cjprofile == 'instexp':
            self.parent.txtPltBox.append('Decay time constant = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        elif self.cjprofile == 'square':
            self.parent.txtPltBox.append('Pulse width = {0:.5g} millisec'
                .format(self.cjargs[3] * 1000))
        self.parent.txtPltBox.append("---\n")

        t, c, Popen, P = cjumps.solve_jump(self.parent.mec, self.cjlen, self.cjstep,
            self.cjfunc, self.cjargs)
        maxP = max(Popen)
        maxC = max(c)
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(t * 1000, c1, 'k-')
        self.parent.canvas.axes.plot(t * 1000, Popen, 'k-')
        for i in range (0, self.parent.mec.kA):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'r--')
        for i in range (self.parent.mec.kA, self.parent.mec.kA + self.parent.mec.kB):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'g--')
        for i in range (self.parent.mec.kA + self.parent.mec.kB, self.parent.mec.k):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'b--')
#        self.axes.set_ylim(0.01, maxC * 1.01)
        #self.axes.set_xlim(5, 30)

        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        self.parent.present_plot = np.vstack((t, Popen, c, P))
#        rcj.printout(self.mec, jpar, output=self.log)

    def onPlotCJumpRiseVConc(self):
        """
        Display plot of weighted on-tau versus concentration. Square pulse only.
        """
        
        self.cjprofile == 'square'

        dialog = CJumpParDlg2(self)
        if dialog.exec_():
            cmin, cmax, self.width = dialog.return_par()

        self.parent.txtPltBox.clear()
        str = ('===== WEIGHTED ON/OFF TAU VERSUS CONCENTRATION =====\n' +
            'Pulse width = {0:.5g} millisec\n'.format(self.width * 1000) +
            'Tau ON - blue solid line.\n' +
            'Tau ON dominant component- red dashed line.\n' +
            'Tau OFF - green solid line.\n' +
            'X axis in mM; Y axis in ms.\n' +
            "---\n")
        self.parent.txtPltBox.append(str)

        c, wton, ton, wtoff, toff  = scpl.conc_jump_on_off_taus_versus_conc_plot(self.parent.mec,
            cmin, cmax, self.width)

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(c, wton,'b-', c, wtoff, 'g-', c, ton[-1], 'r--')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        self.parent.present_plot = np.vstack((c, ton[-1], wton, wtoff))

class CJumpParDlg(QDialog):
    """
    Dialog to input realistic concentration pulse parameters.
    """
    def __init__(self, parent=None, profile='rcj', cjlen=0.05, cjstep=5e-6, cjargs=None):
        super(CJumpParDlg, self).__init__(parent)

        self.profile = profile

        self.reclength = cjlen * 1000 # Record length in ms.
        self.step = cjstep * 1e6 # The sample step in microsec.
        if cjargs:
            self.cmax = cjargs[0] * 1000 # in mM.
            self.cb = cjargs[1] * 1000 # Background concentration in mM.
            if self.profile == 'rcj':
                # 'rcj' profile.
                self.centre = cjargs[2] * 1000 # Pulse centre in ms.
                try:
                    self.rise = cjargs[4] * 1e6 # 10-90% rise time for error function in microsec.
                    self.decay = cjargs[5] * 1e6 # 90-10% decay time for error function in microsec.
                except:
                    self.rise = 200 # 10-90% rise time for error function in microsec.
                    self.decay = 200 # 90-10% decay time for error function in microsec.
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms.
            elif self.profile == 'instexp':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.tdec = cjargs[3] * 1000 # Decay time constant (ms)
            elif self.profile == 'square':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms.    
            elif self.profile == 'square2':
                self.prepulse = cjargs[2] * 1000 # Time before pulse starts (ms)
                self.width = cjargs[3] * 1000 # Pulse halfwidth in ms. 
                self.inter = cjargs[4] * 1000 # Interpulse interval in ms. 
        else:
            # Default values
            self.cmax = 1 # in mM.
            self.cb = 0.0 # Background concentration in mM.
            # 'rcj' profile.
            self.centre = 10 # Pulse centre in ms.
            self.rise = 200 # 10-90% rise time for error function in microsec.
            self.decay = 200 # 90-10% decay time for error function in microsec.
            self.width = 10 # Pulse halfwidth in ms.
            # 'instexp' profile
            self.prepulse = self.reclength / 10.0 # Time before pulse starts (ms)
            self.tdec = 2.5 # Decay time constant (ms)
            self.inter = 10 # Interpulse interval in ms. 

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse concentration (mM):"))
        self.concEdit = QLineEdit(unicode(self.cmax))
        self.concEdit.setMaxLength(12)
        self.connect(self.concEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.concEdit)
        layoutMain.addLayout(layout)



        if self.profile == 'rcj':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse centre position (millisec):"))
            self.centreEdit = QLineEdit(unicode(self.centre))
            self.centreEdit.setMaxLength(12)
            self.connect(self.centreEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.centreEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse 10-90% rise time (microsec):"))
            self.riseEdit = QLineEdit(unicode(self.rise))
            self.riseEdit.setMaxLength(12)
            self.connect(self.riseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.riseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Pulse 90-10% decay time (microsec):"))
            self.decayEdit = QLineEdit(unicode(self.decay))
            self.decayEdit.setMaxLength(12)
            self.connect(self.decayEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.decayEdit)
            layoutMain.addLayout(layout)
            
        elif self.profile == 'instexp':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Decay time constant (ms):"))
            self.decayEdit = QLineEdit(unicode(self.tdec))
            self.decayEdit.setMaxLength(12)
            self.connect(self.decayEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.decayEdit)
            layoutMain.addLayout(layout)

        elif self.profile == 'square':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)
        
        elif self.profile == 'square2':

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Time before pulse (ms):"))
            self.prepulseEdit = QLineEdit(unicode(self.prepulse))
            self.prepulseEdit.setMaxLength(12)
            self.connect(self.prepulseEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.prepulseEdit)
            layoutMain.addLayout(layout)

            layout = QHBoxLayout()
            layout.addWidget(QLabel("Concentration pulse width (millisec):"))
            self.widthEdit = QLineEdit(unicode(self.width))
            self.widthEdit.setMaxLength(12)
            self.connect(self.widthEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.widthEdit)
            layoutMain.addLayout(layout)
            
            layout = QHBoxLayout()
            layout.addWidget(QLabel("Interval between pulses (millisec):"))
            self.interEdit = QLineEdit(unicode(self.inter))
            self.interEdit.setMaxLength(12)
            self.connect(self.interEdit, SIGNAL("editingFinished()"),
                self.on_par_changed)
            layout.addWidget(self.interEdit)
            layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Record length (millisec):"))
        self.reclengthEdit = QLineEdit(unicode(self.reclength))
        self.reclengthEdit.setMaxLength(12)
        self.connect(self.reclengthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.reclengthEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Sampling interval (microsec):"))
        self.stepEdit = QLineEdit(unicode(self.step))
        self.stepEdit.setMaxLength(12)
        self.connect(self.stepEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.stepEdit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Background concentration (mM):"))
        self.bckgrconcEdit = QLineEdit(unicode(self.cb))
        self.bckgrconcEdit.setMaxLength(12)
        self.connect(self.bckgrconcEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.bckgrconcEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Design concentration pulse...")

    def on_par_changed(self):
        """
        """

        self.step = float(self.stepEdit.text()) * 1e-6
        self.reclength = float(self.reclengthEdit.text()) * 0.001
        self.cmax = float(self.concEdit.text()) * 0.001
        self.cb = float(self.bckgrconcEdit.text()) * 0.001

        if self.profile == 'rcj':
            self.centre = float(self.centreEdit.text()) * 0.001
            self.rise = float(self.riseEdit.text()) * 1e-6
            self.decay = float(self.decayEdit.text()) * 1e-6
            self.width = float(self.widthEdit.text()) * 0.001
        elif self.profile == 'instexp':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.tdec = float(self.decayEdit.text()) * 0.001
        elif self.profile == 'square':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.width = float(self.widthEdit.text()) * 0.001
        elif self.profile == 'square2':
            self.prepulse = float(self.prepulseEdit.text()) * 0.001
            self.width = float(self.widthEdit.text()) * 0.001
            self.inter = float(self.interEdit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """

        if self.profile == 'rcj':
            cargs = (self.cmax, self.cb, self.centre, self.width,
                self.rise, self.decay)
            cfunc = cjumps.pulse_erf
        elif self.profile == 'instexp':
            cargs = (self.cmax, self.cb, self.prepulse, self.tdec)
            cfunc = cjumps.pulse_instexp
        elif self.profile == 'square':
            cargs = (self.cmax, self.cb, self.prepulse, self.width)
            cfunc = cjumps.pulse_square
        elif self.profile == 'square2':
            cargs = (self.cmax, self.cb, self.prepulse, self.width, self.inter)
            cfunc = cjumps.pulse_square_paired
        return self.reclength, self.step, cfunc, cargs

class CJumpParDlg2(QDialog):
    """
    Dialog to squatre concentration pulse width and concentration range.
    """
    def __init__(self, parent=None, width = 0.01, cmin=1e-6, cmax=0.001):
        super(CJumpParDlg2, self).__init__(parent)

        self.cmin = cmin * 1000 # in mM.
        self.cmax = cmax * 1000 # in mM.
        self.width = width * 1000 # Pulse halfwidth in ms.

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Square concentration pulse:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Start concentration (mM):"))
        self.conc1Edit = QLineEdit(unicode(self.cmin))
        self.conc1Edit.setMaxLength(12)
        self.connect(self.conc1Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.conc1Edit)
        layoutMain.addLayout(layout)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("End concentration (mM):"))
        self.conc2Edit = QLineEdit(unicode(self.cmax))
        self.conc2Edit.setMaxLength(12)
        self.connect(self.conc2Edit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.conc2Edit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Concentration pulse width (millisec):"))
        self.widthEdit = QLineEdit(unicode(self.width))
        self.widthEdit.setMaxLength(12)
        self.connect(self.widthEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.widthEdit)
        layoutMain.addLayout(layout)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Square concentration pulse pulse...")

    def on_par_changed(self):
        """
        """
        self.cmin = float(self.conc1Edit.text()) * 0.001
        self.cmax = float(self.conc2Edit.text()) * 0.001
        self.width = float(self.widthEdit.text()) * 0.001

    def return_par(self):
        """
        Return parameter dictionary on exit.
        """
        return self.cmin, self.cmax, self.width

class ConcProfileDlg(QDialog):
    """
    """
    def __init__(self, parent=None):
        super(ConcProfileDlg, self).__init__(parent)

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        self.squareRB = QRadioButton("&Square pulse")
        self.square2RB = QRadioButton("&Paired square pulses")
        self.realisticRB = QRadioButton("&Realistic pulse")
        self.realisticRB.setChecked(True)
        self.instexpRB = QRadioButton("&Instantaneous rise and exponential decay")
        
        layoutMain.addWidget(self.squareRB)
        layoutMain.addWidget(self.square2RB)
        layoutMain.addWidget(self.realisticRB)
        layoutMain.addWidget(self.instexpRB)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Choose concentration pulse...")

    def return_par(self):
        if self.instexpRB.isChecked():
            profile = 'instexp'
        elif self.realisticRB.isChecked():
            profile = 'rcj'
        elif self.squareRB.isChecked():
            profile = 'square'
        elif self.square2RB.isChecked():
            profile = 'square2'
        return profile
import numpy as np

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from scalcs import cjumps
from scalcs import sccurves as scpl

from scalcs.gui import myqtcommon


class JumpMenu(QMenu):
    """Menu for concentration-jump calculations."""

    def __init__(self, parent):
        super(JumpMenu, self).__init__(parent)
        self.parent = parent
        self.setTitle('&Jumps')

        # Default pulse — realistic erf profile at 1 mM, 10 ms
        self.pulse = cjumps.ErfPulse(cmax=1e-3, width=10e-3)
        self.cjlen  = 0.05    # recording length (s)
        self.cjstep = 5e-6    # sampling interval (s)

        plotJumpPopenAction = myqtcommon.createAction(self,
            "&Concentration jump: Popen", self.onPlotCJumpPopen)
        plotJumpOccupanciesAction = myqtcommon.createAction(self,
            "&Concentration jump: occupancies",
            self.onPlotCJumpOccupancies)
        plotJumpOnOffTauConc = myqtcommon.createAction(self,
            "&Concentration jump: weighted on/off tau versus concentration",
            self.onPlotCJumpRiseVConc)

        self.addActions([plotJumpPopenAction, plotJumpOccupanciesAction,
            plotJumpOnOffTauConc])

    # ------------------------------------------------------------------
    def _pulse_description(self):
        """Return a short text block describing the current pulse."""
        p = self.pulse
        lines = [
            '===== CONCENTRATION JUMP =====',
            'Concentration profile - green solid line.',
            'Relaxation - blue solid line.',
            '',
            'Concentration pulse profile:',
            'Peak concentration = {:.5g} mM'.format(p.cmax * 1000),
            'Background concentration = {:.5g} mM'.format(p.cb * 1000),
        ]
        if isinstance(p, cjumps.ErfPulse):
            lines += [
                '10-90% rise time  = {:.5g} µs'.format(p.rise  * 1e6),
                '90-10% decay time = {:.5g} µs'.format(p.decay * 1e6),
                'Pulse width       = {:.5g} ms'.format(p.width * 1000),
            ]
        elif isinstance(p, cjumps.InstExpPulse):
            lines.append('Decay time constant = {:.5g} ms'.format(p.tdec * 1000))
        elif isinstance(p, (cjumps.SquarePulse, cjumps.PairedSquarePulse)):
            lines.append('Pulse width = {:.5g} ms'.format(p.width * 1000))
            if isinstance(p, cjumps.PairedSquarePulse):
                lines.append('Interpulse interval = {:.5g} ms'.format(p.inter * 1000))
        lines.append('---')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def onPlotCJumpPopen(self):
        """Display concentration jump Popen time course."""
        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            profile = dialog1.return_par()
        else:
            return

        dialog = CJumpParDlg(self, profile, self.cjlen, self.cjstep, self.pulse)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.pulse = dialog.return_par()
        else:
            return

        self.parent.txtPltBox.clear()
        self.parent.txtPltBox.append(self._pulse_description())

        result = cjumps.solve(self.parent.mec, self.pulse, self.cjlen, self.cjstep)
        t, c, Popen, P = result
        maxP = Popen.max()
        maxC = c.max()
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(t * 1000, Popen, 'b-', t * 1000, c1, 'g-')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        if isinstance(self.pulse, (cjumps.ErfPulse, cjumps.SquarePulse)):
            square = cjumps.SquarePulse(cmax=self.pulse.cmax, width=self.pulse.width,
                                         cb=self.pulse.cb)
            self.parent.log.write(cjumps.printout(self.parent.mec, square))

        self.parent.present_plot = np.vstack((t, Popen, c, P))

    # ------------------------------------------------------------------
    def onPlotCJumpOccupancies(self):
        """Display state occupancy time courses during concentration jump."""
        dialog1 = ConcProfileDlg(self)
        if dialog1.exec_():
            profile = dialog1.return_par()
        else:
            return

        dialog = CJumpParDlg(self, profile, self.cjlen, self.cjstep, self.pulse)
        if dialog.exec_():
            self.cjlen, self.cjstep, self.pulse = dialog.return_par()
        else:
            return

        self.parent.txtPltBox.clear()
        desc = self._pulse_description().replace(
            '===== CONCENTRATION JUMP =====',
            '===== REALISTIC CONCENTRATION JUMP =====\n'
            'Popen relaxation - black solid line.\n'
            'Occupancies of open states - red dashed lines.\n'
            'Occupancies of shortlived shut states - green dashed lines.\n'
            'Occupancies of longlived shut states - blue dashed lines.')
        self.parent.txtPltBox.append(desc)

        result = cjumps.solve(self.parent.mec, self.pulse, self.cjlen, self.cjstep)
        t, c, Popen, P = result
        maxP = Popen.max()
        maxC = c.max()
        c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.plot(t * 1000, c1, 'k-')
        self.parent.canvas.axes.plot(t * 1000, Popen, 'k-')
        for i in range(self.parent.mec.kA):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'r--')
        for i in range(self.parent.mec.kA, self.parent.mec.kA + self.parent.mec.kB):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'g--')
        for i in range(self.parent.mec.kA + self.parent.mec.kB, self.parent.mec.k):
            self.parent.canvas.axes.plot(t * 1000, P[i], 'b--')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        if isinstance(self.pulse, (cjumps.ErfPulse, cjumps.SquarePulse)):
            square = cjumps.SquarePulse(cmax=self.pulse.cmax, width=self.pulse.width,
                                         cb=self.pulse.cb)
            self.parent.log.write(cjumps.printout(self.parent.mec, square))

        self.parent.present_plot = np.vstack((t, Popen, c, P))

    # ------------------------------------------------------------------
    def onPlotCJumpRiseVConc(self):
        """Display plot of weighted on/off tau versus concentration (square pulse)."""
        dialog = CJumpParDlg2(self)
        if not dialog.exec_():
            return
        cmin, cmax, width = dialog.return_par()

        self.parent.txtPltBox.clear()
        self.parent.txtPltBox.append(
            '===== WEIGHTED ON/OFF TAU VERSUS CONCENTRATION =====\n'
            'Pulse width = {:.5g} ms\n'.format(width * 1000) +
            'Tau ON - blue solid line.\n'
            'Tau ON dominant component - red dashed line.\n'
            'Tau OFF - green solid line.\n'
            'X axis in mM; Y axis in ms.\n'
            '---')

        c, wton, ton, wtoff, toff = scpl.conc_jump_on_off_taus_versus_conc_plot(
            self.parent.mec, cmin, cmax, width)

        self.parent.canvas.axes.clear()
        self.parent.canvas.axes.semilogx(c, wton, 'b-', c, wtoff, 'g-', c, ton[-1], 'r--')
        self.parent.canvas.axes.xaxis.set_ticks_position('bottom')
        self.parent.canvas.axes.yaxis.set_ticks_position('left')
        self.parent.canvas.draw()

        self.parent.present_plot = np.vstack((c, ton[-1], wton, wtoff))


# ===========================================================================
# Dialogs
# ===========================================================================

class CJumpParDlg(QDialog):
    """Dialog to set concentration pulse parameters."""

    def __init__(self, parent=None, profile='rcj', cjlen=0.05, cjstep=5e-6,
                 pulse=None):
        super(CJumpParDlg, self).__init__(parent)
        self.profile = profile

        # Convert to display units
        self.reclength = cjlen * 1000       # ms
        self.step      = cjstep * 1e6      # µs

        # Extract parameters from existing pulse, or use defaults
        if pulse is not None and hasattr(pulse, 'cmax'):
            self.cmax = pulse.cmax * 1000   # mM
            self.cb   = pulse.cb   * 1000   # mM
        else:
            self.cmax = 1.0
            self.cb   = 0.0

        # Profile-specific defaults
        if profile == 'rcj':
            if isinstance(pulse, cjumps.ErfPulse):
                self.centre = pulse.centre * 1000   # ms
                self.width  = pulse.width  * 1000
                self.rise   = pulse.rise   * 1e6    # µs
                self.decay  = pulse.decay  * 1e6
            else:
                self.centre = 10.0
                self.width  = 10.0
                self.rise   = 200.0
                self.decay  = 200.0
        elif profile == 'instexp':
            if isinstance(pulse, cjumps.InstExpPulse):
                self.prepulse = pulse.prepulse * 1000
                self.tdec     = pulse.tdec * 1000
            else:
                self.prepulse = 5.0
                self.tdec     = 2.5
        elif profile == 'square':
            if isinstance(pulse, cjumps.SquarePulse):
                self.prepulse = pulse.prepulse * 1000
                self.width    = pulse.width * 1000
            else:
                self.prepulse = 5.0
                self.width    = 10.0
        elif profile == 'square2':
            if isinstance(pulse, cjumps.PairedSquarePulse):
                self.prepulse = pulse.prepulse * 1000
                self.width    = pulse.width * 1000
                self.inter    = pulse.inter * 1000
            else:
                self.prepulse = 5.0
                self.width    = 10.0
                self.inter    = 10.0

        self._build_ui()

    def _build_ui(self):
        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        def row(label, attr, unit_scale=1.0):
            layout = QHBoxLayout()
            layout.addWidget(QLabel(label))
            edit = QLineEdit(str(getattr(self, attr)))
            edit.setMaxLength(12)
            edit.editingFinished.connect(self.on_par_changed)
            layout.addWidget(edit)
            layoutMain.addLayout(layout)
            return edit

        self.concEdit   = row("Pulse concentration (mM):", 'cmax')
        self.bckgrconcEdit = row("Background concentration (mM):", 'cb')

        if self.profile == 'rcj':
            self.widthEdit  = row("Pulse width (ms):", 'width')
            self.centreEdit = row("Pulse centre (ms):", 'centre')
            self.riseEdit   = row("10-90% rise time (µs):", 'rise')
            self.decayEdit  = row("90-10% decay time (µs):", 'decay')
        elif self.profile == 'instexp':
            self.prepulseEdit = row("Time before pulse (ms):", 'prepulse')
            self.decayEdit    = row("Decay time constant (ms):", 'tdec')
        elif self.profile == 'square':
            self.prepulseEdit = row("Time before pulse (ms):", 'prepulse')
            self.widthEdit    = row("Pulse width (ms):", 'width')
        elif self.profile == 'square2':
            self.prepulseEdit = row("Time before pulse (ms):", 'prepulse')
            self.widthEdit    = row("Pulse width (ms):", 'width')
            self.interEdit    = row("Interpulse interval (ms):", 'inter')

        self.reclengthEdit = row("Record length (ms):", 'reclength')
        self.stepEdit      = row("Sampling interval (µs):", 'step')

        layoutMain.addWidget(myqtcommon.ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Design concentration pulse...")

    def on_par_changed(self):
        self.cmax      = float(self.concEdit.text())   * 1e-3
        self.cb        = float(self.bckgrconcEdit.text()) * 1e-3
        self.reclength = float(self.reclengthEdit.text()) * 1e-3
        self.step      = float(self.stepEdit.text())   * 1e-6

        if self.profile == 'rcj':
            self.centre = float(self.centreEdit.text()) * 1e-3
            self.width  = float(self.widthEdit.text())  * 1e-3
            self.rise   = float(self.riseEdit.text())   * 1e-6
            self.decay  = float(self.decayEdit.text())  * 1e-6
        elif self.profile == 'instexp':
            self.prepulse = float(self.prepulseEdit.text()) * 1e-3
            self.tdec     = float(self.decayEdit.text())    * 1e-3
        elif self.profile == 'square':
            self.prepulse = float(self.prepulseEdit.text()) * 1e-3
            self.width    = float(self.widthEdit.text())    * 1e-3
        elif self.profile == 'square2':
            self.prepulse = float(self.prepulseEdit.text()) * 1e-3
            self.width    = float(self.widthEdit.text())    * 1e-3
            self.inter    = float(self.interEdit.text())    * 1e-3

    def return_par(self):
        """Return (reclen, step, pulse) where pulse is a cjumps dataclass."""
        self.on_par_changed()
        if self.profile == 'rcj':
            pulse = cjumps.ErfPulse(
                cmax=self.cmax, width=self.width, cb=self.cb,
                centre=self.centre, rise=self.rise, decay=self.decay)
        elif self.profile == 'instexp':
            pulse = cjumps.InstExpPulse(
                cmax=self.cmax, tdec=self.tdec, cb=self.cb,
                prepulse=self.prepulse)
        elif self.profile == 'square':
            pulse = cjumps.SquarePulse(
                cmax=self.cmax, width=self.width, cb=self.cb,
                prepulse=self.prepulse)
        elif self.profile == 'square2':
            pulse = cjumps.PairedSquarePulse(
                cmax=self.cmax, width=self.width, inter=self.inter, cb=self.cb,
                prepulse=self.prepulse)
        return self.reclength, self.step, pulse


class CJumpParDlg2(QDialog):
    """Dialog for square concentration pulse width and concentration range."""

    def __init__(self, parent=None, width=0.01, cmin=1e-6, cmax=1e-3):
        super(CJumpParDlg2, self).__init__(parent)

        self.cmin  = cmin  * 1000   # mM
        self.cmax  = cmax  * 1000
        self.width = width * 1000   # ms

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Square concentration pulse:"))

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Start concentration (mM):"))
        self.conc1Edit = QLineEdit(str(self.cmin))
        self.conc1Edit.setMaxLength(12)
        self.conc1Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.conc1Edit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("End concentration (mM):"))
        self.conc2Edit = QLineEdit(str(self.cmax))
        self.conc2Edit.setMaxLength(12)
        self.conc2Edit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.conc2Edit)
        layoutMain.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Pulse width (ms):"))
        self.widthEdit = QLineEdit(str(self.width))
        self.widthEdit.setMaxLength(12)
        self.widthEdit.editingFinished.connect(self.on_par_changed)
        layout.addWidget(self.widthEdit)
        layoutMain.addLayout(layout)

        layoutMain.addWidget(myqtcommon.ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Square concentration pulse...")

    def on_par_changed(self):
        self.cmin  = float(self.conc1Edit.text()) * 1e-3
        self.cmax  = float(self.conc2Edit.text()) * 1e-3
        self.width = float(self.widthEdit.text()) * 1e-3

    def return_par(self):
        self.on_par_changed()
        return self.cmin, self.cmax, self.width


class ConcProfileDlg(QDialog):
    """Dialog to choose the concentration pulse shape."""

    def __init__(self, parent=None):
        super(ConcProfileDlg, self).__init__(parent)

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Concentration pulse profile:"))

        self.squareRB    = QRadioButton("&Square pulse")
        self.square2RB   = QRadioButton("&Paired square pulses")
        self.realisticRB = QRadioButton("&Realistic pulse (erf)")
        self.realisticRB.setChecked(True)
        self.instexpRB   = QRadioButton("&Instantaneous rise, exponential decay")

        layoutMain.addWidget(self.squareRB)
        layoutMain.addWidget(self.square2RB)
        layoutMain.addWidget(self.realisticRB)
        layoutMain.addWidget(self.instexpRB)

        layoutMain.addWidget(myqtcommon.ok_cancel_button(self))
        self.setLayout(layoutMain)
        self.setWindowTitle("Choose concentration pulse...")

    def return_par(self):
        if self.instexpRB.isChecked():
            return 'instexp'
        elif self.realisticRB.isChecked():
            return 'rcj'
        elif self.squareRB.isChecked():
            return 'square'
        elif self.square2RB.isChecked():
            return 'square2'

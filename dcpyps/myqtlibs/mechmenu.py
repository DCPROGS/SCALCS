import os
import yaml

try:
    from PySide.QtGui import *
    from PySide.QtCore import *
except:
    raise ImportError("pyqt module is missing")

from dcpyps import dcio
from dcpyps import samples
from dcpyps import mechanism

import myqtcommon

class MechMenu(QMenu):
    """
    """
    def __init__(self, parent):
        super(MechMenu, self).__init__(parent) 
        self.parent = parent
        self.setTitle('&Load Mec')
        
        loadDemo = myqtcommon.createAction(parent, "&Load demo mec",
            self.onLoadDemo)
#        loadDemo2Action = myqtcommon.createAction(parent, "&Load demo: dC-K",
#            self.onLoadDemo_dCK)

        loadFromYAMLFileAction = myqtcommon.createAction(parent,
            "&Load from YAML File...", self.onLoadYAMLmec)
        loadFromMecFileAction = myqtcommon.createAction(parent,
            "&Load from DCprogs MEC File...", self.onLoadMecFile)
        loadFromPrtFileAction = myqtcommon.createAction(parent, 
            "&Load from DCprogs PRT File...", self.onLoadPrtFile)
        loadFromModFileAction = myqtcommon.createAction(parent, 
            "&Load from ChannelLab MOD File...", self.onLoadModFile)
        modifyMecAction = myqtcommon.createAction(parent, 
            "&Modify loaded mec rates", self.onModifyMec)
        modifyStatesAction = myqtcommon.createAction(parent, 
            "&Modify loaded mec states", self.onModifyStates)
        saveMecAction = myqtcommon.createAction(parent, 
            "&Save mec as YAML file...", self.onSaveMec)
            
        self.addActions([loadDemo, #1Action, loadDemo2Action,
            loadFromYAMLFileAction,
            loadFromMecFileAction, loadFromPrtFileAction, loadFromModFileAction,
            modifyMecAction, modifyStatesAction, saveMecAction])
            
    def onLoadDemo(self):
        """
        Load demo mechanisms from dcpyps/samples/samples.py.
        'CH82'- C&H82 numerical example, 'dCK'- delCastillo-Katz mechanism,
        
        """
        dialog = DemoMecDlg()
        if dialog.exec_():
            demo = dialog.return_par() 
        if demo == 'CH82':
            mec = samples.CH82()
        elif demo == 'dCK':
            mec = samples.CCO()
        elif demo == 'CO':
            mec = samples.CO()
        elif demo == 'FCC':
            mec = samples.fully_connected_cycle()
        self.parent.mec = mec
        mec = self.modifyMec(mec, self.parent.log)
        mec.printout(self.parent.log)
        return mec
        
    def onLoadDemo_CH82(self):
        """
        Load demo mechanism (C&H82 numerical example).
        Called from menu Load|Demo.
        """
        self.parent.mec = self.load_demo_mec('CH82', self.parent.log)
        self.parent.log.write("\nLoaded Colquhoun&Hawkes 82 numerical example.\n")
        
    def onLoadDemo_dCK(self):
        """
        Load del Castillo - Katz mechanism.
        """
        self.parent.mec = self.load_demo_mec('dCK', self.parent.log)
        self.parent.log.write("\nLoaded del Castillo-Katz mechanism.\n")
        
    def onLoadMecFile(self):
        """
        Load a mechanism and rates from DC's mec file.
        """
        self.parent.mecfn, filt = QFileDialog.getOpenFileName(self,
            "Open Mec File...", self.parent.path, "DC Mec Files (*.mec *.MEC)")
        self.parent.path = os.path.split(str(self.parent.mecfn))[0]
        self.parent.log.write("\nFile to read: " + 
            os.path.split(str(self.parent.mecfn))[1])

        version, meclist, max_mecnum = dcio.mec_get_list(self.parent.mecfn)
        self.parent.log.write("Mec file version: %d; contains %d mechanisms."
            %(version, max_mecnum))

        dialog = MecListDlg(meclist, max_mecnum, self)
        if dialog.exec_():
            nrate = dialog.returnRates()

        self.parent.mec = dcio.mec_load(self.parent.mecfn, meclist[nrate][0])

        self.modifyMec(self.parent.mec, self.parent.log)

        self.parent.log.write("Loaded mec: " + meclist[nrate][2])
        self.parent.log.write("Loaded rates: " + meclist[nrate][3] + "\n")
        self.parent.mec.printout(self.parent.log)
        
    def onLoadPrtFile(self):
        """
        Load a mechanism and rates from DC's HJCFIT.PRT file.
        """
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open Mec File...", self.parent.path, 
            "DC Mec Files (*.prt *.PRT *.txt *.TXT)")
        self.parent.path = os.path.split(str(filename))[0]
        self.parent.log.write("\nFile to read: " + os.path.split(str(filename))[1])

        self.parent.mec = dcio.mec_load_from_prt(filename)
        self.modifyMec(self.parent.mec, self.parent.log)
        self.parent.log.write("Loaded mec and rates from PRT file: " + filename)
        self.parent.mec.printout(self.parent.log)

    def onLoadModFile(self):
        """
        Load a mechanism and rates from Channel Lab .mod file.
        Called from menu Load|From Channel Lab .MOD File...
        """
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open MOD File...", self.parent.path,
            "Channel Lab MOD Files (*.mod *.MOD)")
        self.parent.path = os.path.split(str(filename))[0]
        self.parent.log.write("\nFile to read: " + 
            os.path.split(str(filename))[1])

        self.parent.mec, title = dcio.mod_load(filename)
        self.modifyMec(self.parent.mec, self.parent.log)
        self.parent.log.write("\n" + title + "\n")
        self.parent.mec.printout(self.parent.log)

    def onModifyMec(self):
        """
        """
        self.parent.mec = self.modifyMec(self.parent.mec, self.parent.log)

    def onModifyStates(self):
        """
        """
        table = StateTableDlg(self, self.parent.mec, self.parent.log)
        if table.exec_():
            self.parent.mec = table.return_mec()

    def onSaveMec(self):
        """
        """

        fname, filt = QFileDialog.getSaveFileName(self,
            "Save mechanism as YAML file...", self.parent.path, ".yaml",
            "YAML files (*.yaml)")
        self.parent.path = os.path.split(str(fname))[0]
        dcio.mec_save_to_yaml(self.parent.mec, fname)
        self.parent.log.write('\n\nMechanism saved in YAML file:')
        self.parent.log.write(fname)

    def onLoadYAMLmec(self):
        """
        """
        filename, filt = QFileDialog.getOpenFileName(self,
            "Open YAML Mec File...", self.parent.path, "YAML Files (*.yaml)")
            
        self.parent.log.write('\n\nLoading mechanism from YAML file:')
        self.parent.log.write(filename)
        stream = file(filename, 'r')
        self.parent.mec = yaml.load(stream)

        self.modifyMec(self.parent.mec, self.parent.log)
        self.parent.log.write("Loaded mec: " + self.parent.mec.mtitle)
        self.parent.log.write("Loaded rates: " + self.parent.mec.rtitle + "\n")
        self.parent.mec.printout(self.parent.log)

    #######################
    def modifyMec(self, mec, out):
        """
        """
        table = RateTableDlg(mec, out)
        if table.exec_():
            mec = table.return_mec()
        mec.printout(out)
        return mec


class MecListDlg(QDialog):
    """
    Dialog to choose mechansim and rates saved in a DC's mec file.
    """
    def __init__(self, meclist, max_mecnum, parent=None):
        super(MecListDlg, self).__init__(parent)

        self.nmec = 1
        self.nrate = 0
        self.meclist = meclist

        self.mList = QListWidget()
        for i in range(1, (max_mecnum + 1)):
                present = False
                id = 0
                for j in range(len(self.meclist)):
                    if i == self.meclist[j][1]:
                        present = True
                        id = j
                if present:
                    self.mList.addItem(str(i) + " "+ self.meclist[id][2])
        self.connect(self.mList,
            SIGNAL("itemSelectionChanged()"),
            self.mecSelected)

        self.rList = QListWidget()
        self.connect(self.rList,
            SIGNAL("itemSelectionChanged()"),
            self.rateSelected)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))

        layout1 = QHBoxLayout()
        layout1.addWidget(self.mList)
        layout1.addWidget(self.rList)
        layout2 = QVBoxLayout()
        layout2.addLayout(layout1)
        layout2.addWidget(buttonBox)

        self.setLayout(layout2)
        self.resize(1000, 500)
        self.setWindowTitle("Choose mechanisms and rates...")

    def mecSelected(self):
        """
        Populate rate list when a mechanism selected.
        """
        self.nmec = self.mList.currentRow() + 1
        self.rList.clear()
        for i in range(len(self.meclist)):
           if self.meclist[i][1] == self.nmec:
               self.rList.addItem(str(i+1) + " "+ self.meclist[i][3])
               self.nrate = i + 1

    def rateSelected(self):
        """
        Get selected rates.
        """
        str1 = self.rList.currentItem().text()
        self.nrate = int(str1.split(" ")[0]) - 1

    def returnRates(self):
        """
        Return rates on exit.
        """
        return self.nrate


class RateTableDlg(QDialog):
    """
    """
    def __init__(self, mec=None, log=None, parent=None):
        super(RateTableDlg, self).__init__(parent)
        self.mec = mec
        self.changed = False
        self.log = log

        layoutMain = QVBoxLayout()
        self.table = RateTable(self.mec)
        self.table.display_data()
        self.connect(self.table,
                SIGNAL("cellChanged(int, int)"),
                self.tableItemChanged)
        layoutMain.addWidget(self.table)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,750,550)
        self.setWindowTitle("View and modify rate constants...")

    def tableItemChanged(self, row, column):
        # TODO: update mechanism if anything changed in table

        if column == 3:
            newratecon = float(self.table.item(row, column).text())
            self.mec.Rates[row].rateconstants = newratecon

        if column == 5: # Fixed rates
            value = False
            if self.table.item(row, column).checkState() > 0:
                value = True
            self.mec.Rates[row].fixed = value
            print 'fixed value=', value

        if column == 6: # MR constrained rates
            value = False
            if self.table.item(row, column).checkState() > 0:
                value = True
                
            if len(self.mec.Cycles) > 1:
                dialog = CycleNumDlg(self, self.mec)
                if dialog.exec_():
                    cyclenum = dialog.return_par()
                self.mec.set_mr(value, row, cyclenum)
            else:
                self.mec.set_mr(value, row)     
                
        if column == 7:
            value = False
            if self.table.item(row, 7).checkState() > 0:
                value = True
            
            if value:
                dialog = ConstrainMultiplyDlg(self, self.mec)
                if dialog.exec_():
                    factor, torate = dialog.return_par()

                self.mec.Rates[row].is_constrained = value
                self.mec.Rates[row].constrain_func = mechanism.constrain_rate_multiple
                self.mec.Rates[row].constrain_args = [torate, factor]

                cell = QTableWidgetItem(str(factor))
                self.table.setItem(row, 8, cell)
                cell = QTableWidgetItem(str(torate+1))
                self.table.setItem(row, 9, cell)         
            else:
                self.mec.Rates[row].is_constrained = value
                self.mec.Rates[row].constrain_func = None
                self.mec.Rates[row].constrain_args = None
                
                cell = QTableWidgetItem('')
                self.table.setItem(row, 8, cell)
                cell = QTableWidgetItem('')
                self.table.setItem(row, 9, cell)

        if column == 10 or column == 11:
            newlimits = [float(self.table.item(row, 10).text()),
                float(self.table.item(row, 11).text())]
            self.mec.Rates[row].limits = newlimits

        self.changed = True

    def return_mec(self):
        if self.changed:
            self.mec.update_constrains()
            self.mec.update_mr()
            self.log.write("\n\nMechanism modified:\n")
#            self.mec.printout(self.log)
        return self.mec

class RateTable(QTableWidget):
    """ Creates a custom table widget """
    def __init__(self, mec=None, *args):
        QTableWidget.__init__(self, *args)
        self.setSelectionMode(self.ContiguousSelection)
        self.setGeometry(0,0,700,400)
        self.setShowGrid(False)
        self.mec = mec

    def display_data(self):
        """ Reads in data as a 2D list and formats and displays it in
            the table """

        header = ['From State', 'To State', 'Rate name', 'Rate value',
            'Conc depend', 'Fixed', 'MR', 'Constr.', 'Factor', 'To rate',
            'Lower limit', 'Higher limit']

        nrows = len(self.mec.Rates)
        ncols = len(header)
        self.setRowCount(nrows)
        self.setColumnCount(ncols)
        self.setHorizontalHeaderLabels(header)

        for i in xrange(nrows):
            cell = QTableWidgetItem(self.mec.Rates[i].State1.name)
            self.setItem(i, 0, cell)
            cell = QTableWidgetItem(self.mec.Rates[i].State2.name)
            self.setItem(i, 1, cell)
            cell = QTableWidgetItem(self.mec.Rates[i].name)
            self.setItem(i, 2, cell)
            cell = QTableWidgetItem(str(self.mec.Rates[i].unit_rate()))
            self.setItem(i, 3, cell)

            if self.mec.Rates[i].effectors[0] is None:
                eff = ''
            else:
                eff = self.mec.Rates[i].effectors[0]
            cell = QTableWidgetItem(eff)
            self.setItem(i, 4, cell)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].fixed > 0:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 5, check)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].mr:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 6, check)

            check = QTableWidgetItem()
            value = Qt.Unchecked
            if self.mec.Rates[i].is_constrained:
                value = Qt.Checked
            check.setCheckState(value)
            self.setItem(i, 7, check)
            factor = ''
            torate = ''
            if self.mec.Rates[i].constrain_args:
                factor = self.mec.Rates[i].constrain_args[1]
                torate = self.mec.Rates[i].constrain_args[0] + 1
            cell = QTableWidgetItem(str(factor))
            self.setItem(i, 8, cell)
            cell = QTableWidgetItem(str(torate))
            self.setItem(i, 9, cell)

            if len(self.mec.Rates[i].limits) == 0:
                if eff == '':
                    limits = [[1e-3,1e+7]]
                else:
                    limits = [[1e-3,1e+10]]
            else:
                limits = self.mec.Rates[i].limits
            cell = QTableWidgetItem(str(limits[0][0]))
            self.setItem(i, 10, cell)
            cell = QTableWidgetItem(str(limits[0][1]))
            self.setItem(i, 11, cell)


        self.resizeColumnsToContents()
        self.resizeRowsToContents()

class StateTableDlg(QDialog):
    """
    """
    def __init__(self, parent=None, mec=None, log=None):
        super(StateTableDlg, self).__init__(parent)
        self.mec = mec
        self.changed = False
        self.log = log

        layoutMain = QVBoxLayout()
        self.table = StateTable(self.mec)
        self.table.display_data()
        self.connect(self.table,
                SIGNAL("cellChanged(int, int)"),
                self.tableItemChanged)
        layoutMain.addWidget(self.table)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setGeometry(100,100,450,350)
        self.setWindowTitle("View and modify states...")

    def tableItemChanged(self, row, column):
        # TODO: update mechanism if anything changed in table

#        if column == 0:
#            newname = self.table.item(row, column).text()
#            self.mec.States[row].name = newname

        if column == 1:
            newtype = self.table.item(row, column).text().upper()
            self.mec.States[row].statetype = newtype

        if column == 2:
            newgamma = float(self.table.item(row, column).text())
            self.mec.Rates[row].conductance = newgamma

        self.changed = True

    def return_mec(self):
        if self.changed:
            self.mec.update_states()
            self.log.write("\n\nMechanism states modified:\n")
            self.mec.printout(self.log)
        return self.mec

class StateTable(QTableWidget):
    """ Creates a custom table widget """
    def __init__(self, mec=None, *args):
        QTableWidget.__init__(self, *args)
        self.setSelectionMode(self.ContiguousSelection)
        self.setGeometry(0,0,700,400)
        self.setShowGrid(False)
        self.mec = mec

    def display_data(self):
        """ Reads in data as a 2D list and formats and displays it in
            the table """

        header = ['State name', 'State Class', 'Conductance',
            'Number of connections']

        nrows = len(self.mec.States)
        ncols = len(header)
        self.setRowCount(nrows)
        self.setColumnCount(ncols)
        self.setHorizontalHeaderLabels(header)

        for i in xrange(nrows):
            cell = QTableWidgetItem(self.mec.States[i].name)
            self.setItem(i, 0, cell)
            cell = QTableWidgetItem(self.mec.States[i].statetype)
            self.setItem(i, 1, cell)
            cell = QTableWidgetItem(str(self.mec.States[i].conductance))
            self.setItem(i, 2, cell)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class CycleNumDlg(QDialog):
    """
    """
    def __init__(self, parent=None, mec=None):
        super(CycleNumDlg, self).__init__(parent)

        self.mec = mec

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Check cycle to which proposed rate belongs:"))

        self.cyclesRB = []
        for cycle in self.mec.Cycles:
            states = "Cycle: " + ' - '.join(cycle.states)
            self.cyclesRB.append(QRadioButton(states))
        self.cyclesRB[0].setChecked(True)
       
        for cycle in self.cyclesRB:
            layoutMain.addWidget(cycle)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Choose cycle...")

    def return_par(self):
        cyclenum = None
        for i in range(len(self.cyclesRB)):
            if self.cyclesRB[i].isChecked():
                cyclenum = i
        return cyclenum
    
class ConstrainMultiplyDlg(QDialog):
    """
    """
    def __init__(self, parent=None, mec=None):
        super(ConstrainMultiplyDlg, self).__init__(parent)

        self.mec = mec
        self.factor = 1
        layoutMain = QVBoxLayout()
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Factor:"))
        self.fEdit = QLineEdit(unicode(self.factor))
        self.fEdit.setMaxLength(12)
        self.connect(self.fEdit, SIGNAL("editingFinished()"),
            self.on_par_changed)
        layout.addWidget(self.fEdit)
        layoutMain.addLayout(layout)
        
        layoutMain.addWidget(QLabel("Check rate:"))
        self.ratesRB = []
        counter = 1
        for rate in self.mec.Rates:
            name = "{0:d}: {1}".format(counter, rate.name)
            self.ratesRB.append(QRadioButton(name))
            counter += 1
        self.ratesRB[0].setChecked(True)
        for rate in self.ratesRB:
            layoutMain.addWidget(rate)
            
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)
        self.setLayout(layoutMain)
        self.setWindowTitle("Specify factor and rate...")
        
    def on_par_changed(self):
        self.factor = float(self.fEdit.text())

    def return_par(self):
        torate = None
        for i in range(len(self.ratesRB)):
            if self.ratesRB[i].isChecked():
                torate = i
        return self.factor, torate    
    

class DemoMecDlg(QDialog):
    """
    """
    def __init__(self, parent=None):
        super(DemoMecDlg, self).__init__(parent)

        layoutMain = QVBoxLayout()
        layoutMain.addWidget(QLabel("Demo mechanisms:"))

        self.CH82 = QRadioButton("&CH82")
        self.CH82.setChecked(True)
        self.dCK = QRadioButton("&del Castillo - Katz")
        self.CO = QRadioButton("&C(losed)-O(pen)")
        self.FCC = QRadioButton("&Fully connected cycle")
        
        layoutMain.addWidget(self.CH82)
        layoutMain.addWidget(self.dCK)
        layoutMain.addWidget(self.CO)
        layoutMain.addWidget(self.FCC)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok|
            QDialogButtonBox.Cancel)
        self.connect(buttonBox, SIGNAL("accepted()"),
            self, SLOT("accept()"))
        self.connect(buttonBox, SIGNAL("rejected()"),
            self, SLOT("reject()"))
        layoutMain.addWidget(buttonBox)

        self.setLayout(layoutMain)
        self.setWindowTitle("Choose demo mechanism...")
        
    def return_par(self):
        if self.CH82.isChecked():
            mec = 'CH82'
        elif self.CO.isChecked():
            mec = 'CO'
        elif self.dCK.isChecked():
            mec = 'dCK'
        elif self.FCC.isChecked():
            mec = 'FCC'
        return mec
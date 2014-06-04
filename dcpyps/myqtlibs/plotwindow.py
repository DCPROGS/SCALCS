try:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import scale as mscale
    from matplotlib import transforms as mtransforms
    from matplotlib import ticker
except:
    raise ImportError("matplotlib module is missing")

class MatPlotWin(FigureCanvas):
    """
    """
    def __init__(self, size=(6.0, 4.0), fsize=8):
        # Prepare matplotlib plot window
        self.fig = Figure(size, dpi=85)
        self.axes = self.fig.add_subplot(111)
        self.axes.autoscale_view(True,True,True)
        self.fontsize = fsize
        for loc, spine in self.axes.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        for label in self.axes.xaxis.get_ticklabels():
            label.set_fontsize(self.fontsize)
        for label in self.axes.yaxis.get_ticklabels():
            label.set_fontsize(self.fontsize)
#        self.mplTools = NavigationToolbar(self.canvas, self.parent)
        mscale.register_scale(SquareRootScale)
        FigureCanvas.__init__(self, self.fig)        
        
class MatPlotTools(NavigationToolbar):
    """
    """
    def __init__(self, canvas, parent):
        NavigationToolbar.__init__(self, canvas, parent)
        
        
class SquareRootScale(mscale.ScaleBase):
    """
    Class for generating square root scaled axis for probability density
    function plots.
    """

    name = 'sqrtscale'
    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
    def get_transform(self):
        """
        Set the actual transform for the axis coordinates.
        """
        return self.SqrTransform()
    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults.
        """
        axis.set_major_formatter(ticker.ScalarFormatter())

    class SqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            """
            Take numpy array and return transformed copy.
            """
            return np.sqrt(a)
        def inverted(self):
            """
            Get inverse transform.
            """
            return SquareRootScale.InvertedSqrTransform()

    class InvertedSqrTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)
        def transform(self, a):
            return np.power(a, 2)
        def inverted(self):
            return SquareRootScale.SqrTransform()

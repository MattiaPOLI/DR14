{
  import plotly
  
  def enable_plotly_in_cell():
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    plotly.offline.init_notebook_mode(connected=True) 
}

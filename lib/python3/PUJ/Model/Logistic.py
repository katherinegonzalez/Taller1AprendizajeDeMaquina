## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import *

'''
'''
class Logistic( Linear ):

  ## -----------------------------------------------------------------------
  ## Initialize an object witha zero-sized parameters vector
  ## -----------------------------------------------------------------------
  def __init__( self ):
    pass
  # end if

  ## -----------------------------------------------------------------------
  ## Final-user methods.
  ## -----------------------------------------------------------------------
  def evaluate( self, x ):
    return 1.0 / ( 1.0 + numpy.exp( -super( ).evaluate( x ) ) )
  # end def

  def threshold( self, x ):
    z = self.evaluate( x )
    return ( z >= 0.5 ).astype( z.dtype )
  # end def

  ## -----------------------------------------------------------------------
  '''
  Cross cost function for linear regressions.
  '''
  class Cost( Base.Cost ):

    ## ---------------------------------------------------------------------
    ## Initialize an object witha zero-sized parameters vector
    ## ---------------------------------------------------------------------
    def __init__( self, model, X, y ):
      super( ).__init__( model, X, y )
      self.m_mY = y.mean( )
      self.m_XhY = numpy.matrix( numpy.multiply( X, y ).mean( axis = 0 ) ).T
    # end def

    ## ---------------------------------------------------------------------
    ## Evaluate cost with gradient (if needed)
    ## ---------------------------------------------------------------------
    def evaluate( self, need_gradient = False ):
      z = self.m_Model.evaluate( self.m_X )
      J  = numpy.log( z[ self.m_Y == 1 ] + 1e-12 ).sum( )
      J += numpy.log( 1 - z[ self.m_Y == 0 ] + 1e-12 ).sum( )
      J /= -float( self.m_X.shape[ 0 ] )
            

      ### Nueva parte del codigo
      ### Cresta
      valor_lambda = 100
      
      g = self.m_Model.parameters( )
      ww = g[ 1 : , : ]
      bb = g[ 0 , 0 ]
      
      J_Cres = valor_lambda * (ww.T@ww+(bb*bb))
      J += J_Cres 

      ### Fin nueva parte del codigo
      
      if need_gradient:

        g = numpy.zeros( self.m_Model.parameters( ).shape )
        g[ 0 , 0 ] = z.mean( ) - self.m_mY + 2*valor_lambda*g[ 0 , 0 ]
        g[ 1 : , : ] = \
          numpy.matrix( numpy.multiply( self.m_X, z ) ).mean( axis = 0 ).T - \
          self.m_XhY + 2*valor_lambda*g[ 1 : , : ].sum( )
        return [ J, g ]
      else:
        return [ J, None ]
      
      # end if
    # end def
  # end class
# end class

## eof - $RCSfile$

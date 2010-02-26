#ifndef MANTID_GEOMETRY_FITPARAMETER_H_
#define MANTID_GEOMETRY_FITPARAMETER_H_

//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
//#include <fstream>
#include "MantidKernel/System.h"

namespace Mantid
{
  namespace Geometry
  {
    /**
    Store information about a fitting parameter such as its value
    if it is constrained or tied. Main purpose is to use this 
    class as a type for storing information about a fitting parameter
    in the parameter map of the workspace.

    @author Anders Markvardsen, ISIS, RAL
    @date 26/2/2010

    Copyright &copy; 2007-10 STFC Rutherford Appleton Laboratory

    This file is part of Mantid.

    Mantid is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    Mantid is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    File change history is stored at: <https://svn.mantidproject.org/mantid/trunk/Code/Mantid>
    Code Documentation is available at: <http://doxygen.mantidproject.org>
    */
    class DLLExport FitParameter 
    {
    public:
      /// Constructor
      FitParameter() {};

      /// get paramter value
      double getValue() const {return m_value;}
      /// set parameter value
      double& setValue() {return m_value;}
      /// get tie
      std::string getTie() { return m_tie; }
      /// set tie
      std::string& setTie() { return m_tie; }

      /// Prints object to stream
      void printSelf(std::ostream& os) const;

    private:
      /// value of parameter
      double m_value;
      /// tie of parameter
      std::string m_tie;
    };

    // Overload operator <<
    DLLExport std::ostream& operator<<(std::ostream&, const FitParameter& );
    DLLExport std::istream& operator>>(std::istream&,FitParameter&);

  } // namespace Geometry
} // namespace Mantid

#endif /*MANTID_GEOMETRY_FITPARAMETER_H_*/

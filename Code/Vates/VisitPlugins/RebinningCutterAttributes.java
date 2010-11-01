// ***************************************************************************
//
// Copyright (c) 2000 - 2010, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory
// LLNL-CODE-442911
// All rights reserved.
//
// This file is  part of VisIt. For  details, see https://visit.llnl.gov/.  The
// full copyright notice is contained in the file COPYRIGHT located at the root
// of the VisIt distribution or at http://www.llnl.gov/visit/copyright.html.
//
// Redistribution  and  use  in  source  and  binary  forms,  with  or  without
// modification, are permitted provided that the following conditions are met:
//
//  - Redistributions of  source code must  retain the above  copyright notice,
//    this list of conditions and the disclaimer below.
//  - Redistributions in binary form must reproduce the above copyright notice,
//    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
//    documentation and/or other materials provided with the distribution.
//  - Neither the name of  the LLNS/LLNL nor the names of  its contributors may
//    be used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
// ARE  DISCLAIMED. IN  NO EVENT  SHALL LAWRENCE  LIVERMORE NATIONAL  SECURITY,
// LLC, THE  U.S.  DEPARTMENT OF  ENERGY  OR  CONTRIBUTORS BE  LIABLE  FOR  ANY
// DIRECT,  INDIRECT,   INCIDENTAL,   SPECIAL,   EXEMPLARY,  OR   CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
// SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
// CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
// LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
// OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
//
// ***************************************************************************

package llnl.visit.operators;

import llnl.visit.AttributeSubject;
import llnl.visit.CommunicationBuffer;
import llnl.visit.Plugin;
import java.lang.Double;
import java.util.Vector;

// ****************************************************************************
// Class: RebinningCutterAttributes
//
// Purpose:
//    Attributes for the Rebinning Cutter
//
// Notes:      Autogenerated by xml2java.
//
// Programmer: xml2java
// Creation:   omitted
//
// Modifications:
//   
// ****************************************************************************

public class RebinningCutterAttributes extends AttributeSubject implements Plugin
{
    private static int RebinningCutterAttributes_numAdditionalAtts = 2;

    public RebinningCutterAttributes()
    {
        super(RebinningCutterAttributes_numAdditionalAtts);

        origin = new Vector();
        normal = new Vector();
    }

    public RebinningCutterAttributes(int nMoreFields)
    {
        super(RebinningCutterAttributes_numAdditionalAtts + nMoreFields);

        origin = new Vector();
        normal = new Vector();
    }

    public RebinningCutterAttributes(RebinningCutterAttributes obj)
    {
        super(RebinningCutterAttributes_numAdditionalAtts);

        int i;

        origin = new Vector(obj.origin.size());
        for(i = 0; i < obj.origin.size(); ++i)
        {
            Double dv = (Double)obj.origin.elementAt(i);
            origin.addElement(new Double(dv.doubleValue()));
        }

        normal = new Vector(obj.normal.size());
        for(i = 0; i < obj.normal.size(); ++i)
        {
            Double dv = (Double)obj.normal.elementAt(i);
            normal.addElement(new Double(dv.doubleValue()));
        }


        SelectAll();
    }

    public int Offset()
    {
        return super.Offset() + super.GetNumAdditionalAttributes();
    }

    public int GetNumAdditionalAttributes()
    {
        return RebinningCutterAttributes_numAdditionalAtts;
    }

    public boolean equals(RebinningCutterAttributes obj)
    {
        int i;

        // Compare the elements in the origin vector.
        boolean origin_equal = (obj.origin.size() == origin.size());
        for(i = 0; (i < origin.size()) && origin_equal; ++i)
        {
            // Make references to Double from Object.
            Double origin1 = (Double)origin.elementAt(i);
            Double origin2 = (Double)obj.origin.elementAt(i);
            origin_equal = origin1.equals(origin2);
        }
        // Compare the elements in the normal vector.
        boolean normal_equal = (obj.normal.size() == normal.size());
        for(i = 0; (i < normal.size()) && normal_equal; ++i)
        {
            // Make references to Double from Object.
            Double normal1 = (Double)normal.elementAt(i);
            Double normal2 = (Double)obj.normal.elementAt(i);
            normal_equal = normal1.equals(normal2);
        }
        // Create the return value
        return (origin_equal &&
                normal_equal);
    }

    public String GetName() { return "RebinningCutter"; }
    public String GetVersion() { return "1.0"; }

    // Property setting methods
    public void SetOrigin(Vector origin_)
    {
        origin = origin_;
        Select(0);
    }

    public void SetNormal(Vector normal_)
    {
        normal = normal_;
        Select(1);
    }

    // Property getting methods
    public Vector GetOrigin() { return origin; }
    public Vector GetNormal() { return normal; }

    // Write and read methods.
    public void WriteAtts(CommunicationBuffer buf)
    {
        if(WriteSelect(0, buf))
            buf.WriteDoubleVector(origin);
        if(WriteSelect(1, buf))
            buf.WriteDoubleVector(normal);
    }

    public void ReadAtts(int index, CommunicationBuffer buf)
    {
        switch(index)
        {
        case 0:
            SetOrigin(buf.ReadDoubleVector());
            break;
        case 1:
            SetNormal(buf.ReadDoubleVector());
            break;
        }
    }

    public String toString(String indent)
    {
        String str = new String();
        str = str + doubleVectorToString("origin", origin, indent) + "\n";
        str = str + doubleVectorToString("normal", normal, indent) + "\n";
        return str;
    }


    // Attributes
    private Vector origin; // vector of Double objects
    private Vector normal; // vector of Double objects
}


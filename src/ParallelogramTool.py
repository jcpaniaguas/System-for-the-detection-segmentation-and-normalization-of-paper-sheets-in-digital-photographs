# @author: jcpaniaguas
import math


class ParallelogramTool:
    """Class with geometric functions for a parallelogram.
    """

    @staticmethod
    def parallelogram_rule(a,b,c):
        """Parallelogram rule. With three given points calculate where the
        fourth point would be, taking into account that it will be in front of B.

        Args:
            a ([(int,int)]): Point A.
            b ([(int,int)]): Point B.
            c ([(int,int)]): Point C.

        Returns:
            [(int,int)]: Point D. The point opposite B.
        """
        a1,a2 = a
        b1,b2 = b
        c1,c2 = c
        return a1+c1-b1,a2+c2-b2
    
    @staticmethod
    def angle(a,b,c):
        """Three points are passed and A is selected returning alpha, the angle between AB and AC.

        Args:
            a ([(int,int)]): Point A. Where the angle is located.
            b ([(int,int)]): Point B.
            c ([(int,int)]): Point C.

        Returns:
            [float]: Degrees of angle alpha, between AB and AC.
        """
        AB = (b[0]-a[0],b[1]-a[1])
        AC = (c[0]-a[0],c[1]-a[1])
        scalar = AB[0]*AC[0]+AB[1]*AC[1]
        modAB = math.sqrt((AB[0]**2) + (AB[1]**2))
        modAC = math.sqrt((AC[0]**2) + (AC[1]**2))
        alpha = math.acos(scalar/(modAB*modAC))
        return math.degrees(alpha)
    
    @staticmethod
    def distance(A,B):
        """Distance between two points.

        Args:
            A ([(int,int)]): Point A.
            B ([(int,int)]): Point B.

        Returns:
            [float]: Distance between A and B.
        """
        return math.sqrt(((A[0]-B[0])**2)+((A[1]-B[1])**2))

    @staticmethod
    def vector(A,B):
        """Obtain the vector AB.

        Args:
            A ([(int,int)]): Point A.
            B ([(int,int)]): Point B.

        Returns:
            [(int,int)]: Vector AB.
        """
        return (B[0]-A[0],B[1]-A[1])

    @staticmethod
    def analyze_distance(a):
        """Analyze the possible parallelogram.

        Args:
            a ([[(int,int)]]): List of the four points.

        Returns:
            [dict()]: Dictionary analyzing the four points. Returns:
                Point A
                Point B
                Point C
                Point D
                Distance AB 
                Distance BD
                Distance DC
                Distance CA
                Half of AD segment
                Half of BC segment
                Angle ABC
                Angle DBC
                Angle CDA
                Angle BDA
                Perimeter 
        """
        found = False
        a = sorted(a, key=lambda p: float(p[1]))
        AB = a[0:2]
        CD = a[2:]
        AB = sorted(AB, key=lambda p: float(p[0]))
        CD = sorted(CD, key=lambda p: float(p[0]))
        A = AB[0]
        B = AB[1]
        C = CD[0]
        D = CD[1]
        A = (int(A[0]),int(A[1]))
        B = (int(B[0]),int(B[1]))
        C = (int(C[0]),int(C[1]))
        D = (int(D[0]),int(D[1]))
        distAB = ParallelogramTool.distance(A,B)
        distBD = ParallelogramTool.distance(B,D)
        distDC = ParallelogramTool.distance(D,C)
        distCA = ParallelogramTool.distance(C,A)
        distBC = ParallelogramTool.distance(B,C)
        distAD = ParallelogramTool.distance(A,D)
        distBC = ParallelogramTool.distance(B,C)
        ABC = ParallelogramTool.angle(A,B,C)
        DBC = ParallelogramTool.angle(D,B,C)
        CDA = ParallelogramTool.angle(C,D,A)
        BDA = ParallelogramTool.angle(B,D,A)
        perimeter = 2*distAB*distBC
        d = dict()
        d['A'] = A
        d['B'] = B
        d['C'] = C
        d['D'] = D
        d['distAB'] = distAB 
        d['distBD'] = distBD
        d['distDC'] = distDC
        d['distCA'] = distCA
        d['halfAD'] = int((A[0]+D[0])/2),int((A[1]+D[1])/2)
        d['halfBC'] = int((B[0]+C[0])/2),int((B[1]+C[1])/2)
        d['ABC'] = ABC
        d['DBC'] = DBC
        d['CDA'] = CDA
        d['BDA'] = BDA
        d['perimeter'] = perimeter 
        return d

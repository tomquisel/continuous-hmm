package edu.epfl.naef.hmm;

/** Base class for emission values */
abstract class Emission{}

/** Emission wrapper for a double. */
class RealEmission extends Emission{
    double v;

    RealEmission(double v){
        this.v = v;
    }

    double val(){return v;}

    public String toString(){return ""+v;}
}

/** Models an Emission which is a vector of doubles. */
class VectorEmission extends Emission{
    double v[];

    VectorEmission(double[] v){
        this.v = v;
    }

    double[] val(){return v;}

    RealEmission elem(int i){return new RealEmission(v[i]);}

    public String toString(){
        String s="<";
        String fmt="%.03f";

        if(v.length>0)
            s+=String.format(fmt,v[0]);
        for(int i=1;i<v.length;i++){
            s+=","+String.format(fmt,v[i]);
        }
        return s+">";
    }
}


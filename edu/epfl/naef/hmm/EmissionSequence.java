package edu.epfl.naef.hmm;

import java.io.*;
import java.util.*;

/** container class for a sequence of emissions. */
class EmissionSequence{
    
    /** An EmissionSequence is just an array of Emissions. */
    Emission[] es;

    /** length of es */
    public int length;
    
    EmissionSequence(Emission[] es){
        this.es = es;
        length = es.length;
    }

    Emission[] val(){
        return es;
    }

    /** convert a list of doubles into an EmissionSequence of RealEmissions.
     * A useful static function for creating EmissionSequences.
     * @param xs array of doubles to convert
     * @return an EmissionSequence representing xs
     */
    static EmissionSequence toRealSeq(double[] xs){
        RealEmission[] rs = new RealEmission[xs.length];
        for(int i=0;i<xs.length;i++){
            rs[i] = new RealEmission(xs[i]);
        }
        return new EmissionSequence(rs);
    }

    /** convert a matrix of doubles into an EmissionSequence of VectorEmissions
    * xs[t][i] becomes the ith element of the vector at time t
    */
    static EmissionSequence toVectorSeq(double[][] xs){
        VectorEmission[] vs = new VectorEmission[xs.length];
        for(int i=0;i<xs.length;i++){
            vs[i] = new VectorEmission(xs[i]);
        }
        return new EmissionSequence(vs);
    }

    static EmissionSequence readFromFile(String filename) throws Exception{
        return readFromFile(filename,false);
    }
    
    /** read an EmissionSequence of vectors in from a datafile.
     * The datafile must be in the fou output format or an alternate
     * reduced format. 
     * @param filename name of the file to read
     * @param labeled is the file labeled? no longer needed...
     * @return the EmissionSequence read from file
     */
    static EmissionSequence readFromFile(String filename,
            boolean labeled) throws Exception{
        ArrayList<VectorEmission> vectors = new ArrayList();
        ArrayList<Integer> labels = new ArrayList();
        BufferedReader file = new BufferedReader(new FileReader(filename));
        String line;
        int num=1;
        
        while((line = file.readLine()) != null){
            // skip blank lines
            if(line.matches("^\\s*$"))
                continue;

            String[] fields = line.split("\\s+");

            // decided if this file is labeled or not
            if(num == 1){
                if(fields.length == 15 || fields.length == 4){
                    labeled = true;
                }
            }
            
            double[] vector= new double[2];
            // ensure there are the right number of fields on the line        
            if((!labeled && fields.length == 14) || fields.length == 15){
                // make a vector out of the relevent pieces of the line
                //vector[0] = Double.parseDouble(fields[10]);
                // use the p-value calculated by fou instead of amplitude
                vector[0] = Double.parseDouble(fields[12]);
                vector[1] = Double.parseDouble(fields[11]);
                if(labeled)
                    labels.add(Integer.parseInt(fields[14]));
            }
            // handle another file format
            else if(fields.length == 4){
                // make a vector out of the relevent pieces of the line
                vector[0] = Double.parseDouble(fields[1]);
                vector[1] = Double.parseDouble(fields[2]);
                if(labeled)
                    labels.add(Integer.parseInt(fields[3]));
            }
            else{
                System.out.println("File Format Error on line "+
                        num+": Expected 4, 14, or 15 fields, found "+fields.length);
                file.close();
                System.exit(0);
            }

            // add the current emission to vectors
            vectors.add(new VectorEmission(vector));
            num++;
        }

        file.close();
        
        VectorEmission[] vea = new VectorEmission[vectors.size()];
        if(labeled){
            int[] labelarray = new int[labels.size()];
            // stupid manual unboxing, even in 1.5 GARH
            for(int i=0;i<labels.size();i++){
                labelarray[i] = labels.get(i);
            }
            return new LabeledEmissionSequence(vectors.toArray(vea),
                                               labelarray);
        }
        else
            return new EmissionSequence(vectors.toArray(vea));
    }

    void print(){
        for(int i=0;i<length;i++){
            System.out.println(es[i].toString());
        }
    }
}

/** holds an emission sequence along with state labels used for training. */
class LabeledEmissionSequence extends EmissionSequence {
    int[] labels;
    int nstates;
    
    LabeledEmissionSequence(EmissionSequence es, int[] labels){
        this(es.es,labels);
    }
    
    LabeledEmissionSequence(Emission[] es, int[] labels){
        super(es);

        // mismatched emission & label arrays
        if(labels.length != length){
            System.out.println("Attempt to construct a LabeledEmissionSequence\n"
                    +" from "+es.length+" emissions and "+
                    labels.length+" labels");
            System.exit(0);
        }
        this.labels = labels;

        // states are labeled 0 to N-1.
        // determine the number of states by finding the largest state label
        int highest = 0;
        for(int i=0;i<length;i++){
            if(labels[i] > highest){
                highest = labels[i];
            }
        }
        this.nstates = highest + 1;
    }

    int[] getLabels(){return labels;}
    int getNumStates(){return nstates;}
}

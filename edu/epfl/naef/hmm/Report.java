package edu.epfl.naef.hmm;

/** Print pretty reports of HMM results. */
class Report{
    static void printReport(HMM hmm, EmissionSequence es){
        printReport(hmm,es,true,false);
    }

    /** print a summary of an HMM and the posterior probabilities of an
     * EmissionSequence.
     *
     * @param hmm HMM to print summary information about
     * @param es print the posterior decoding of this EmissionSequence 
     * @param withTable do we want to show the per-emission results?
     * @param withLabels should the per-emission results have labels?
     */
    static void printReport(HMM hmm, EmissionSequence es, boolean withTable, 
            boolean withLabels){
        
        // recalculate forwards and backwards tables
        Forward fwd = new Forward(hmm, es);
        Backward bwd = new Backward(hmm, es);
        double logP = fwd.logprob();
        Viterbi vit = new Viterbi(hmm,es);
        String path = vit.getPath();

        // if there are labels, and we want to display them, display them
        LabeledEmissionSequence les = null;
        if(withLabels && (es instanceof LabeledEmissionSequence)){
            les = (LabeledEmissionSequence) es;
        }


        if(withTable){
            System.out.println("----------------------------------");
            // print out explanatory table
            String[] headers = {"time","best_path"};
            String[] states = hmm.getStateNames();
            String fmt = "%-11s";
            String efmt = "%-16s";
            // print header information
            for(int i=0;i<1;i++)
                System.out.printf(fmt,headers[i]);
            System.out.printf(efmt,"emission");
            for(int i=1;i<states.length;i++)
                System.out.printf(efmt,"emission_"+states[i]);
            System.out.print("| ");
            for(int i=1;i<states.length;i++)
                System.out.printf(fmt,"emitprob_"+states[i]);
            System.out.print("| ");
            for(int i=1;i<states.length;i++)
                System.out.printf(fmt,"statprob_"+states[i]);
            System.out.print("| ");
            for(int i=1;i<headers.length;i++)
                System.out.printf(fmt,headers[i]);
            // print out label header
            if(les != null){
                System.out.print("| label");
            }
            System.out.println();

            // calculate column information
            Distribution[] dists = hmm.getDistributions();

            Emission[] xs = es.val();

            String fl = "%-10";
            String efl = "%-15";
            // print column information
            for(int i=0;i<es.length;i++){
                System.out.printf(fl+"d ",i);
                System.out.printf(efl+"s ",xs[i]);
                for(int j=1;j<dists.length;j++)
                    System.out.printf(efl+"s ",dists[j].getTransformedInput(es,i));
                System.out.print("| ");
                for(int j=1;j<dists.length;j++)
                    System.out.printf(fl+"f ",dists[j].pdf(es,i));
                System.out.print("| ");
                for(int j=1;j<dists.length;j++)
                    System.out.printf(fl+"f ",
                            Math.exp(fwd.f[i+1][j]+bwd.b[i+1][j]-logP));
                System.out.print("| ");
                System.out.printf(fl+"s ",path.charAt(i));
                // print out label for this probe
                if(les != null){
                    //System.out.print("| "+((les.getLabels()[i] == 1)?"E":"N"));
                    System.out.print("| "+les.getLabels()[i]);
                }
                System.out.println();
            }
        }

        // print summary information
        System.out.println("----------------------------------");
        System.out.println("Best fit HMM for the data:");
        hmm.print(new SystemOut());
        System.out.println("----------------------------------");
        System.out.println("Log probability of data given the HMM:");
        System.out.println(logP);

        // print the best sequence
        System.out.println("----------------------------------");
        System.out.println("Best Solution:");
        System.out.println(path);
    }

}


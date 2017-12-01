// Modified 2006 by Thomas Quisel to handle continue distributions

// Implementation of some algorithms for pairwise alignment from
// Durbin et al: Biological Sequence Analysis, CUP 1998, chapter 3.
// Peter Sestoft, sestoft@dina.kvl.dk (Oct 1999), 2001-08-20 version 0.7
// Reference:  http://www.dina.kvl.dk/~sestoft/bsa.html

// License: Anybody can use this code for any purpose, including
// teaching, research, and commercial purposes, provided proper
// reference is made to its origin.  Neither the author nor the Royal
// Veterinary and Agricultural University, Copenhagen, Denmark, can
// take any responsibility for the consequences of using this code.

// Compile with:
//      javac Match3.java
// Run with:
//      java Match3


// Notational conventions: 

// i     = 1,...,L           indexes x, the observed string, x_0 not a symbol
// k,ell = 0,...,hmm.nstate-1  indexes hmm.state(k)   a_0 is the start state

package edu.epfl.naef.hmm;

import java.text.*;
import java.util.*;
import java.io.*;

// Some algorithms for Hidden Markov Models (Chapter 3): Viterbi,
// Forward, Backward, Baum-Welch.  We compute with log probabilities.

class HMM {
    // State names and state-to-state transition probabilities
    int nstate;           // number of states (incl initial state)
    String[] state;       // names of the states
    double[][] loga;      // loga[k][ell] = log(P(k -> ell))

    // Emission names and emission probabilities
    //int nesym;            // number of emission symbols
    //String esym;          // the emission symbols e1,...,eL (characters)
    //double[][] loge;      // loge[k][ei] = log(P(emit ei in state k))

    /** loge[k] represents the continuous, real valued distribution of outputs
     * in state k.  */
    Distribution[] loge; 

    /** Create an HMM from state and transition information.
     * @param state array of state names (except intial state)
     * @param amat matrix of transition probabilities (except initial state)
     * @param emat  matrix of emission probabilities
     */
    public HMM(String[] state, double[][] amat, 
            Distribution[] emat) {
        if (state.length != amat.length)
            throw new IllegalArgumentException("HMM: state and amat disagree");
        if (amat.length != emat.length)
            throw new IllegalArgumentException("HMM: amat and emat disagree");
        for (int i=0; i<amat.length; i++) {
            if (state.length != amat[i].length)
                throw new IllegalArgumentException("HMM: amat non-square");
        }      

        // Set up the transition matrix
        nstate = state.length + 1;
        this.state = new String[nstate];
        loga = new double[nstate][nstate];
        this.state[0] = "B";        // initial state
        // P(start -> start) = 0
        loga[0][0] = Double.NEGATIVE_INFINITY; // = log(0)
        // P(start -> other) = 1.0/state.length 
        double fromstart = Math.log(1.0/state.length);
        for (int j=1; j<nstate; j++)
            loga[0][j] = fromstart;
        for (int i=1; i<nstate; i++) {
            // Reverse state names for efficient backwards concatenation
            this.state[i] = new StringBuffer(state[i-1]).reverse().toString();
            // P(other -> start) = 0
            loga[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
            for (int j=1; j<nstate; j++)
                loga[i][j] = Math.log(amat[i-1][j-1]);
        }

        // Set up the emission matrix
        loge = new Distribution[nstate];
        loge[0] = new Null();
        for(int i=1;i<nstate;i++){
            loge[i] = emat[i-1]; 
        }
    }

    public Distribution[] getDistributions(){
        return loge;
    }

    public String[] getStateNames(){
        return state;
    }

    public void print(Output out) { printa(out); printe(out); }

    public void printa(Output out) {
        out.println("Transition probabilities:");

        // print first line
        out.print("  ");
        for(int i=1;i<nstate;i++)
            out.print(state[i]+hdrpad);
        out.println();

        // print table
        for (int i=1; i<nstate; i++) {
            out.print(state[i]+" ");
            for (int j=1; j<nstate; j++) 
                out.print(fmtlog(loga[i][j]));
            out.println();
        }
    }

    public void printe(Output out) {
        out.println("Emission probabilities:");
        for (int i=1; i<nstate; i++) {
            out.println(state[i]+": "+loge[i].toString());
        }
    }

    private static DecimalFormat fmt = new DecimalFormat("0.000000 ");
    private static String hdrpad     =                    "        ";

    public static String fmtlog(double x) {
        if (x == Double.NEGATIVE_INFINITY)
            return fmt.format(0);
        else
            return fmt.format(Math.exp(x));
    }

    /** Trains initial HMM parameters using labeled training, and then
     * applies Baum Welch to refine the model.
     *
     * @param les LabeledEmissionSequence to train the HMM on
     * @param state state names for the new HMM
     * @param emat emission probabilities for the HMM states
     * @return the fully trained HMM
     */
    public static HMM smartStart(LabeledEmissionSequence[] les,
            String[] state,Distribution[] emat){
        HMM start = labeledTrain(les,state,emat);
        return baumwelch(les,start,500);
    }
    
    public static HMM baumwelch(EmissionSequence[] es, HMM hmm,
            final double threshold) {
        return baumwelch(es,hmm,null,null,threshold);
    }

    public static HMM baumwelch(EmissionSequence[] es, String[] state,
            Distribution[] emat,
            final double threshold) {
        return baumwelch(es,null,state,emat,threshold);
    }

    /** The Baum-Welch algorithm for estimating HMM parameters for a
     * given model topology and a family of observed sequences.
     * Often gets stuck at a non-global minimum; depends on initial guess.
     * 
     * @param es set of EmissionSequences to train on
     * @param hmm specifies the initial HMM to start the EM loop
     * @param state if hmm is unspecified, the list of state names
     * @param emat if hmm is unspecified, the list of emission distributions
     * @param threshold if the loglikelihood changes by less than this, STOP
     * @return the trained HMM
     */
    public static HMM baumwelch(EmissionSequence[] es, HMM hmm, String[] state,
            Distribution[] emat,
            final double threshold) {
        int nseqs  = es.length;
        Forward[] fwds = new Forward[nseqs];
        Backward[] bwds = new Backward[nseqs];
        double[] logP = new double[nseqs];
        int nstate;

        // properly initialize the number of states
        if(state != null)
            nstate = state.length;
        else
            nstate = hmm.getStateNames().length-1;
            
        double[][] amat = new double[nstate][];

        // Initially use random transition and emission matrices
        for (int k=0; k<nstate; k++) {
            amat[k] = randomdiscrete(nstate);
        }

        if(hmm == null){
            hmm = new HMM(state, amat, emat);
        }
        else{
            state = new String[nstate];
            for(int i=0;i<nstate;i++)
                state[i] = hmm.getStateNames()[i+1];
            emat = new Distribution[nstate];
            for(int i=0;i<nstate;i++)
                emat[i] = hmm.getDistributions()[i+1];
        }

        double oldloglikelihood;

        // Compute Forward and Backward tables for the sequences
        double loglikelihood = fwdbwd(hmm, es, fwds, bwds, logP);
        System.out.println("log likelihood = " + loglikelihood);
        int count = 1;
        do {
            oldloglikelihood = loglikelihood;
            // Compute estimates for A and E
            double[][] A = new double[nstate][nstate];
            //Distribution[] E = new Distribution[nstate];

            // The Model works by transitioning, and then emitting.
            // Thus, the very first state does not emit anything.

            for (int s=0; s<nseqs; s++) {
                EmissionSequence e = es[s];
                Forward fwd  = fwds[s];
                Backward bwd = bwds[s];
                int L = e.length;
                double P = logP[s];	// NOT exp.  Fixed 2001-08-20

                // estimate new parameters for transition probabilities
                for (int i=0; i<L-1; i++) 
                    for (int k=0; k<nstate; k++) 
                        for (int ell=0; ell<nstate; ell++){
                            A[k][ell] += exp(fwd.f[i+1][k+1] 
                                    + hmm.loga[k+1][ell+1] 
                                    + hmm.loge[ell+1].lpdf(e,i+1)
                                    + bwd.b[i+2][ell+1] 
                                    - P);
                        }

            }

            // estimate new parameters for output probabilities
            for(int i=0;i<nstate;i++){
                emat[i].estimateParams(es,logP,i,fwds,bwds);
            }
            
            
            // normalized transition probabilities
            for (int k=0; k<nstate; k++) {
                double Aksum = 0;
                for (int ell=0; ell<nstate; ell++)
                    Aksum += A[k][ell];
                for (int ell=0; ell<nstate; ell++)
                    amat[k][ell] = A[k][ell] / Aksum;
            }

            // Create new model 
            hmm = new HMM(state, amat, emat);//E);
            // write the HMM parameters and the distributions they were
            // estimated from to file.
            writeFeatureDistribution("/home/trq/hmm/system/hmm/hmmdistrib.step"
                    + count,fwds[0], bwds[0],hmm.getDistributions(),es[0]);

            loglikelihood = fwdbwd(hmm, es, fwds, bwds, logP);
            System.out.println("log likelihood = " + loglikelihood);
            count++;
        } while (Math.abs(oldloglikelihood - loglikelihood) > threshold);
        return hmm;
    }

    /** Train HMM parameters based on labels in the training data. This
     * resembles Baum Welch, but it does not iterate. Parameters are estimated
     * only once to maximize the likelihood of the model given the labels.
     *
     * @param les LabeledEmissionSequence to train the HMM on
     * @param state array of state names for the new HMM
     * @param emat array of emission distributions for the new HMM
     * @return an HMM trained on les
     */
    public static HMM labeledTrain(LabeledEmissionSequence[] les,
            String[] state,Distribution[] emat){
        int nstate = state.length;
        int nseqs  = les.length;

        Forward[] fwds = new Forward[nseqs];
        Backward[] bwds = new Backward[nseqs];
        double[] logP = new double[nseqs];

        double[][] amat = new double[nstate][nstate];

        HMM hmm = new HMM(state, amat, emat);

        double loglikelihood=1;

        // Compute Forward and Backward tables for the sequences
        for(int i=0;i<nseqs;i++){
            fwds[i] = Forward.makeFromLabeled(hmm,les[i]);
            bwds[i] = Backward.makeFromLabeled(hmm,les[i]);
        }

        // Compute estimates for A and E
        double[][] A = new double[nstate][nstate];

        // The Model works by transitioning, and then emitting.
        // Thus, the very first state does not emit anything.

        for (int s=0; s<nseqs; s++) {
            EmissionSequence e = les[s];
            Forward fwd  = fwds[s];
            Backward bwd = bwds[s];
            int L = e.length;
            double P = logP[s];	// NOT exp.  Fixed 2001-08-20

            // estimate new parameters for transition probabilities
            for (int i=0; i<L-1; i++) 
                for (int k=0; k<nstate; k++) 
                    for (int ell=0; ell<nstate; ell++){
                        if(fwd.f[i+1][k+1] == 0 && bwd.b[i+2][ell+1] == 0){
                            A[k][ell] += 1;
                        }
                    }
        }

        // estimate new parameters for output probabilities
        for(int i=0;i<nstate;i++){
            emat[i].estimateParams(les,logP,i,fwds,bwds);
        }


        // normalized transition probabilities
        for (int k=0; k<nstate; k++) {
            double Aksum = 0;
            for (int ell=0; ell<nstate; ell++){
                Aksum += A[k][ell];
                //System.out.println("A["+k+"]["+ell+"] = "+A[k][ell]);
            }
            for (int ell=0; ell<nstate; ell++)
                amat[k][ell] = A[k][ell] / Aksum;
        }

        // Create new model 
        hmm = new HMM(state, amat, emat);

        writeFeatureDistribution("/home/trq/hmm/system/hmm/hmmdistrib.step0",fwds[0],
                bwds[0],hmm.getDistributions(),les[0]);

        loglikelihood = fwdbwd(hmm, les, fwds, bwds, logP);
        System.out.println("log likelihood = " + loglikelihood);

        return hmm;
    }

    /** Calculates the forward and backward tables for an HMM and 
     * EmissionSequence[] pair.
     *
     * @param hmm HMM to use
     * @param es EmissionSequence to use
     * @param fwds the resulting forward tables are returned in this argument
     * @param bwds the resulting forward tables are returned in this argument
     * @param logP the loglikelihood of the HMM given each sequence is put here
     * @return the overall logliklihood of the HMM
     */
    private static double fwdbwd(HMM hmm, EmissionSequence[] es, Forward[] fwds, 
            Backward[] bwds, double[] logP) {
        double loglikelihood = 0;
        for (int s=0; s<es.length; s++) {
            fwds[s] = new Forward(hmm, es[s]);
            bwds[s] = new Backward(hmm, es[s]);
            logP[s] = fwds[s].logprob();
            loglikelihood += logP[s];
        }
        return loglikelihood;
    }


    // This is a hackish addin to allow visualization of the model training
    // process. It should really go in its own class. It dumps the current
    // model parameters and a histogram of the emissions they are trying to
    // model to file.
    
    // number of buckets in the histogram
    public static final int nbkt = 100;
    // minimum value to be included in the histogram
    public static final double minval = 0;
    // max value in the histogram
    public static final double maxval = 30;
    /** Writes model parameters and an emission histogram to file.
     * @param file file to write the data to
     * @param fwd forward table for the EmissionSequence so we can calculate state probabilities
     * @param bwd backward table, same use as fwd
     * @param dists distributions so we can parameters and transform the data
     * @param es EmissionSequence to calculate the histogram over
     * @return void
     */
    public static void writeFeatureDistribution(String file, Forward fwd,
            Backward bwd, Distribution[] dists, EmissionSequence es)
    {
        double domain = maxval - minval;
        double width = domain / nbkt;
        double logP = fwd.logprob();
        try{
            PrintWriter out = new PrintWriter(new FileWriter(file));
            
            // write out model parameters (for the Gamma distribution, anyway)
            for(int i=1;i<dists.length;i++){
                VectorDist dist = (VectorDist) dists[i];
                double[] params = dist.getFirstGammaParams();
                out.println(params[0] + " " + params[1]);
            }
            out.println();

            // calculate weight that goes in each bucket
            for(int i=1;i<dists.length;i++){
                double[] buckets = new double[nbkt];
                for(int j=0;j<es.length;j++){
                    // calculate emission value from es
                    double logpval = ((VectorEmission)dists[i].
                            getTransformedInput(es,j)).val()[0];
                    // find which bin this output corresponds to
                    int bin = (int)((logpval - minval) / width);
                    // increment this bin's weight by the probability of being
                    // in state i at time j
                    buckets[bin] += Math.exp(fwd.f[j+1][i]+bwd.b[j+1][i]-logP);
                }

                // print out the bucket weights
                for(int j=0;j<buckets.length;j++){
                    out.println(buckets[j]);
                }
                out.println();
            }
            out.close();
        } catch(IOException e){
            System.out.println("Error writing histogram to "+file);
        }
    }

    public static double exp(double x) {
        if (x == Double.NEGATIVE_INFINITY)
            return 0;
        else
            return Math.exp(x);
    }

    private static double[] uniformdiscrete(int n) {
        double[] ps = new double[n];
        for (int i=0; i<n; i++) 
            ps[i] = 1.0/n;
        return ps;
    }    

    private static double[] randomdiscrete(int n) {
        double[] ps = new double[n];
        double sum = 0;
        // Generate random numbers
        for (int i=0; i<n; i++) {
            ps[i] = Math.random();
            sum += ps[i];
        }
        // Scale to obtain a discrete probability distribution
        for (int i=0; i<n; i++) 
            ps[i] /= sum;
        return ps;
    }
}


/** Abstract class for algorithms that operate on an HMM, EMissionSequence pair*/
abstract class HMMAlgo {
    HMM hmm;                      // the hidden Markov model
    EmissionSequence x;                     // the observed string of emissions

    public HMMAlgo(HMM hmm, EmissionSequence x) 
    { this.hmm = hmm; this.x = x; }

    // Compute log(p+q) from plog = log p and qlog = log q, using that
    // log (p + q) = log (p(1 + q/p)) = log p + log(1 + q/p) 
    // = log p + log(1 + exp(log q - log p)) = plog + log(1 + exp(logq - logp))
    // and that log(1 + exp(d)) < 1E-17 for d < -37.

    static double logplus(double plog, double qlog) {
        double max, diff;
        //System.out.println("Plog:"+plog+" QLog: "+qlog);
        if (plog > qlog) {
            if (qlog == Double.NEGATIVE_INFINITY)
                return plog;
            else {
                max = plog; diff = qlog - plog;
            } 
        } else {
            if (plog == Double.NEGATIVE_INFINITY)
                return qlog;
            else {
                max = qlog; diff = plog - qlog;
            }
        }
        //System.out.println(" Diff: "+diff);
        // Now diff <= 0 so Math.exp(diff) will not overflow
        return max + (diff < -37 ? 0 : Math.log(1 + Math.exp(diff)));
    }

    public int length(){return x.length;}
}

class TestLogPlus {
    // Test HMMAlgo.logplus: it passes these tests
    public static void main(String[] args) {
        final double EPS = 1E-14;
        Random rnd = new Random();
        int count = Integer.parseInt(args[0]);
        for (int k=200; k>=-200; k--) 
            for (int i=0; i<count; i++) {
                double logp = Math.abs(rnd.nextDouble()) * Math.pow(10, k);
                double logq = Math.abs(rnd.nextDouble());
                double logpplusq = HMMAlgo.logplus(logp, logq);
                double p = Math.exp(logp), 
                       q = Math.exp(logq), 
                       pplusq = Math.exp(logpplusq);
                if (Math.abs(p+q-pplusq) > EPS * pplusq) 
                    System.out.println(p + "+" + q + "-" + pplusq);
            }
    }
}

/** The Viterbi algorithm: find the most probable state path producing
 * the observed outputs x */
class Viterbi extends HMMAlgo {
    double[][] v;         // the matrix used to find the decoding
    // v[i][k] = v_k(i) = 
    // log(max(P(pi in state k has sym i | path pi)))
    Traceback2[][] B;     // the traceback matrix
    Traceback2 B0;        // the start of the traceback 

    public Viterbi(HMM hmm, EmissionSequence x) {
        super(hmm, x);
        final int L = x.length;
        v = new double[L+1][hmm.nstate];
        B = new Traceback2[L+1][hmm.nstate];
        v[0][0] = 0;                // = log(1)
        for (int k=1; k<hmm.nstate; k++)
            v[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i=1; i<=L; i++)
            v[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i=1; i<=L; i++)
            for (int ell=0; ell<hmm.nstate; ell++) {
                int kmax = 0;
                double maxprod = v[i-1][kmax] + hmm.loga[kmax][ell];
                for (int k=1; k<hmm.nstate; k++) {
                    double prod = v[i-1][k] + hmm.loga[k][ell];
                    if (prod > maxprod) {
                        kmax = k;
                        maxprod = prod;
                    }
                }
                v[i][ell] = hmm.loge[ell].lpdf(x,i-1) + maxprod;
                B[i][ell] = new Traceback2(i-1, kmax);
            }
        int kmax = 0;
        double max = v[L][kmax];
        for (int k=1; k<hmm.nstate; k++) {
            if (v[L][k] > max) {
                kmax = k;
                max = v[L][k];
            }
        }
        B0 = new Traceback2(L, kmax);
    }

    public String getPath() {
        StringBuffer res = new StringBuffer();
        Traceback2 tb = B0;
        int i = tb.i, j = tb.j;
        while ((tb = B[tb.i][tb.j]) != null) {
            res.append(hmm.state[j]);
            i = tb.i; j = tb.j;
        }        
        return res.reverse().toString();
    }

    public void print(Output out) {
        for (int j=0; j<hmm.nstate; j++) {
            for (int i=0; i<v.length; i++)
                out.print(HMM.fmtlog(v[i][j]));
            out.println();
        }
    }
}


/** The `Forward algorithm': find the probability of an observed sequence x */
class Forward extends HMMAlgo {
    double[][] f;                 // the matrix used to find the decoding
    // f[i][k] = f_k(i) = log(P(x1..xi, pi_i=k))
    private int L; 

    public Forward(HMM hmm, EmissionSequence x) {
        super(hmm, x);
        L = x.length;
        f = new double[L+1][hmm.nstate];
        f[0][0] = 0;                // = log(1)
        for (int k=1; k<hmm.nstate; k++)
            f[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i=1; i<=L; i++)
            f[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i=1; i<=L; i++)
            for (int ell=1; ell<hmm.nstate; ell++) {
                double sum = Double.NEGATIVE_INFINITY; // = log(0)
                for (int k=0; k<hmm.nstate; k++){
                    sum = logplus(sum, f[i-1][k] + hmm.loga[k][ell]);
                }
                //System.out.println("Nstates:"+hmm.nstate+" L:"+L+" ell:"+ell+" i:"+i+" logelen:"+hmm.loge.length);
                f[i][ell] = hmm.loge[ell].lpdf(x,i-1) + sum;
            }
    }

    public static Forward makeFromLabeled(HMM hmm, LabeledEmissionSequence les){
        int L = les.length;
        int nstates = les.getNumStates()+1;
        double[][] f = new double[L+1][nstates];
        int[] labels = les.getLabels();
        for(int i=1;i<=L;i++){
            for(int j=0;j<nstates;j++){
                f[i][j] = Double.NEGATIVE_INFINITY;
            }
            f[i][labels[i-1]+1]=0;
        }
        Forward fwd = new Forward(hmm,les);
        fwd.f = f;
        return fwd;
    }

    double logprob() {
        double sum = Double.NEGATIVE_INFINITY; // = log(0)
        for (int k=0; k<hmm.nstate; k++) 
            sum = logplus(sum, f[L][k]);
        return sum;
    }

    public void print(Output out) {
        for (int j=0; j<hmm.nstate; j++) {
            for (int i=0; i<f.length; i++)
                out.print(HMM.fmtlog(f[i][j]));
            out.println();
        }
    }
}


/** The `Backward algorithm': find the probability of an observed sequence x */
class Backward extends HMMAlgo {
    double[][] b;               // the matrix used to find the decoding
    // b[i][k] = b_k(i) = log(P(x(i+1)..xL, pi_i=k))

    public Backward(HMM hmm, EmissionSequence x) {
        super(hmm, x);
        int L = x.length;
        b = new double[L+1][hmm.nstate];

        //***** [trq] ADDED... I think this is a bug to not set b[L][0] ****
        //***** however, b[L][0] will never have any impact because proba[*][k] = 0
        b[L][0]=Double.NEGATIVE_INFINITY; // = log(0)

        for (int k=1; k<hmm.nstate; k++)
            b[L][k] = 0;// = log(1)
        for (int i=L-1; i>=1; i--)
            for (int k=0; k<hmm.nstate; k++) {
                double sum = Double.NEGATIVE_INFINITY; // = log(0)
                for (int ell=1; ell<hmm.nstate; ell++) 
                    sum = logplus(sum, hmm.loga[k][ell] 
                            + hmm.loge[ell].lpdf(x,i)
                            + b[i+1][ell]);
                b[i][k] = sum;
            }
    }

    public static Backward makeFromLabeled(HMM hmm, LabeledEmissionSequence les){
        int L = les.length;
        int nstates = les.getNumStates()+1;
        double[][] b = new double[L+1][nstates];
        int[] labels = les.getLabels();
        for(int i=1;i<=L;i++){
            for(int j=0;j<nstates;j++){
                b[i][j] = Double.NEGATIVE_INFINITY;
            }
            b[i][labels[i-1]+1]=0;
        }
        Backward bwd = new Backward(hmm,les);
        bwd.b = b;
        return bwd;
    }

    double logprob() {
        double sum = Double.NEGATIVE_INFINITY; // = log(0)
        for (int ell=0; ell<hmm.nstate; ell++) 
            sum = logplus(sum, hmm.loga[0][ell] 
                    + hmm.loge[ell].lpdf(x,0)
                    + b[1][ell]);
        return sum;
    }

    public void print(Output out) {
        for (int j=0; j<hmm.nstate; j++) {
            for (int i=0; i<b.length; i++)
                out.print(HMM.fmtlog(b[i][j]));
            out.println();
        }
    }
}


/** Compute posterior probabilities using Forward and Backward */
class PosteriorProb {
    Forward fwd;                  // result of the forward algorithm 
    Backward bwd;                 // result of the backward algorithm 
    private double logprob;

    PosteriorProb(Forward fwd, Backward bwd) {
        this.fwd = fwd; this.bwd = bwd;
        logprob = fwd.logprob();    // should equal bwd.logprob()
    }

    double posterior(int i, int k) // i=index into the seq; k=the HMM state
    { return Math.exp(fwd.f[i][k] + bwd.b[i][k] - logprob); }
}


// Traceback objects

abstract class Traceback {
    int i, j;                     // absolute coordinates
}


// Traceback2 objects

class Traceback2 extends Traceback {
    public Traceback2(int i, int j)
    { this.i = i; this.j = j; }
}


// Auxiliary classes for output

abstract class Output {
    public abstract void print(String s);
    public abstract void println(String s);
    public abstract void println();
}

class SystemOut extends Output {
    public void print(String s)
    { System.out.print(s); }

    public void println(String s)
    { System.out.println(s); }

    public void println()
    { System.out.println(); }
}


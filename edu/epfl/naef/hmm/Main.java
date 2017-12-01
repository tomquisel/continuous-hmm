package edu.epfl.naef.hmm;

//import org.rosuda.JRI.Rengine;
//import org.rosuda.JRI.REXP;

public class Main {
    //Rengine re;
    
    public static void main(String[] args) throws Exception{
        // initialize R engine - used by the HMM
        //Rengine re=new Rengine(args, false, null);
        // the engine creates R is a new thread, so we should wait until it's ready
        /*if (!re.waitForR()) {
            System.out.println("Cannot load R");
            return;
        }*/

        //expTest();
        //normTest();
        //diffTest();
        //CpG();
        //gammaTest();
        //finalTest();
        runFromArgs(args);
    }

    static void badArgs(){
        System.out.println("Incorrect Arguments. Possible Formats:\n"+
"java Main [args] file.txt\n"+
"   Runs Baum-Welch on a single file and reports posterior probabilities\n"+
"java Main -labeled [args] train.txt test.txt\n"+
"   Runs labeled training on train.txt and reports posterior probs on test.txt\n"+
"java Main -smart [args] train.txt test.txt\n"+
"   Runs labeled then Baum Welch training on train.txt and\n"+
"   reports posterior probs on test.txt\n"+
"java Main -test [args]\n"+
"   Runs labeled training on some hard coded test data and reports results\n"+
"   Used for testing the sanity of labeled training\n"+
"java Main -bw [args] train.txt test.txt\n"+
"   Runs Baum-Welch on train.txt and reports posterior probs on test.txt\n"+
"   Used to test the generalizability of a Baum-Welch trained model\n"+
"Arguments:\n"+
"-wn {n floats}\n"+
"   Used to indicate the weights on n emission components\n"+
"   Example: -w2 1.2 0.0\n"+
"-l Print out labels if they are available\n"+
"");
        System.exit(0);
    }

    /** stores weights which determine how much a particular component of an
     * HMM emission vector affects the vector's emission probability
     */
    static double weights[];
    /** If we're using labeled HMM training, do we want to print those labels?
     */
    static boolean printLabels;
    
    /** Processes command line options. 
     * @param args the array of arguments
     * @param pos the index into args where option processing should start
     * @return the position of the first unprocessed argument
     */
    static int processArgs(String[] args, int pos){
        int num = 0;

        // initialize variables controlled by arguments
        weights = null;
        printLabels = false;
        
        while(pos < args.length){
            // is there a list of weights specified?
            if(args[pos].matches("-w\\d+")){
                // if so, find out how long the list is
                String[] strs = args[pos].split("-w");
                try{
                    num = Integer.parseInt(strs[1]);
                }catch(Exception e){}

                if(num < 1) badArgs();

                // populate the weights list
                weights = new double[num];
                for(int i=0;i<num;i++){
                    try{
                        weights[i] = Double.parseDouble(args[pos+1+i]);
                    }catch(Exception e){badArgs();}
                }
                pos += 1 + num;
            }
            else if(args[pos].equals("-l")){
                printLabels = true;
                pos++;
            }
            else{
                // if no argument matches, stop looking for arguments
                break;
            }
        }

        // return the position of the first unprocessed argument
        return pos;
    }
    
    /** runs one of several different HMM systems based on the arguments.
     * It throws an exception because I'd rather have most exceptions bubble
     * up to the toplevel for debugging rather than writing lots of repetitive
     * exception handlers.
     */
    static void runFromArgs(String[] args) throws Exception{
        int type=0;
        if(args.length == 0)
            badArgs();

        int pos;

        if(args[0].equals("-labeled")){
            pos = processArgs(args,1);
            if(pos+2 != args.length) badArgs();
            trainAndEval(args[pos],args[pos+1]);
        }
        else if(args[0].equals("-smart")){
            pos = processArgs(args,1);
            if(pos+2 != args.length) badArgs();
            smart(args[pos],args[pos+1]);
        }
        else if(args[0].equals("-test")){
            pos = processArgs(args,1);
            if(pos != args.length) badArgs();
            labeledTest();
        }
        else if(args[0].equals("-bw")){
            pos = processArgs(args,1);
            if(pos+2 != args.length) badArgs();
            BWtrainAndEval(args[pos],args[pos+1]);
        }
        else{
            pos = processArgs(args,0);
            if(pos+1 != args.length) badArgs();
            runOnFile(args[pos]);
        }
    }

    /** The original function to return the emission distrubution for the HMM*/
    static Distribution[] getOldEProbs(){
        ZeroGuard zeroguard = new ZeroGuard(0.001);
        //                       A ~ G(a,t)   phi ~ U(0,2pi)
        Distribution[] Ndists = {/*new Exp(20)*/Gamma.rand(zeroguard),new Uniform(0,24)};
        //                       A ~ G(a,t)
        Distribution[] Edists = {/*new CenteredNorm(0.7,0.001)*/Gamma.rand(zeroguard),
        //  phi_i - phi_{i-1} ~ N(0,d^2)
            new CenteredNorm(0,/*10*/Math.random()*2+1,new CircDistTransform(24))};

        Distribution[] eprob = { /*N*/ new VectorDist(Ndists,weights),
                                 /*E*/ new VectorDist(Edists,weights)};
        return eprob;
    }

    /** the newer, more experimental set of emission distributions for the HMM*/
    static Distribution[] getEProbs(){
        ZeroGuard neglogpval = new ZeroGuard(0.001,new NegLogTransform());
        //ZeroGuard neglogpval = new ZeroGuard(0.001,new NegLogTransform(/*new ZeroGuard(0.001,new OneMinusTransform())*/));
        //OneMinusTransform neglogpval = new OneMinusTransform(new ZeroGuard(0.001/*,new NegLogTransform()*/));
        CircDistTransform phasediff = new CircDistTransform(24);
        //                       A ~ G(a,t)   phi ~ U(0,2pi)
        Distribution[] Ndists = {/*new Exp(20)*/Gamma.rand(neglogpval),new CenteredNorm(0,/*10*/Math.random()*2+1,phasediff)};
        //                       A ~ G(a,t)
        Distribution[] Edists = {/*new CenteredNorm(0.7,0.001)*/Gamma.rand(neglogpval),
        //  phi_i - phi_{i-1} ~ N(0,d^2)
            new CenteredNorm(0,/*10*/Math.random()*2+1,phasediff)};

        Distribution[] eprob = { /*N*/ new VectorDist(Ndists,weights),
                                 /*E*/ new VectorDist(Edists,weights)};
        return eprob;
    }
    
    /** Runs the 2-state HMM with the emission distributions from getEProbs()
     * on data from a file. 
     * 
     * @param file file to run the HMM on
     */
    static void runOnFile(String file) throws Exception{
        String[] state = { "N", "E" };
        Distribution[] eprob = getEProbs();
 
        EmissionSequence[] es = {EmissionSequence.readFromFile(file)};
        //System.out.println("----------------------------------");
        //System.out.println("Sequence:");
        //es[0].print();
        //System.exit(0);
        
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.1);
        Report.printReport(estimate,es[0],true,printLabels);
    }

    /** Trains the standard HMM on trainfile, uses it to decode evalfile. 
     * The twist is that the HMM's initial weights are trained using labeled
     * training on trainfile, and then they are refined using Baum Welch on 
     * trainfile. Once the HMM is fully trained, its estimated posterior 
     * probabilities for the data in evalfile are printed out.
     *
     * @param trainfile labeled data file to train HMM
     * @param evalfile unlabeled data file on which to do posterior decoding
     */
    static void smart(String trainfile,String evalfile) throws Exception{
        String[] state = { "N", "E" };
        Distribution[] eprob = getEProbs();
 
        // train the model
        LabeledEmissionSequence[] les = 
        {(LabeledEmissionSequence)EmissionSequence.readFromFile(trainfile)};
        HMM model = HMM.smartStart(les,state,eprob);

        // evaluate the trained model on the evaluation data
        Report.printReport(model,EmissionSequence.readFromFile(evalfile),true,printLabels);
    }
    
    /** Trains the standard HMM on trainfile, uses it to decode evalfile. Training
     * is done using non-iterative labeled training.
     * 
     * @param trainfile labeled data file to train HMM
     * @param evalfile unlabeled data file on which to do posterior decoding
     */
    static void trainAndEval(String trainfile,String evalfile) throws Exception{
        String[] state = { "N", "E" };
        Distribution[] eprob = getEProbs();
 
        // train the model
        LabeledEmissionSequence[] les = 
        {(LabeledEmissionSequence)EmissionSequence.readFromFile(trainfile)};
        HMM model = HMM.labeledTrain(les,state,eprob);

        // evaluate the trained model on the evaluation data
        Report.printReport(model,EmissionSequence.readFromFile(evalfile),true,printLabels);
    }

    /** Trains the standard HMM on trainfile, uses it to decode evalfile. Training
     * is done using the Baum Welch algorithm. Used to test how well the models
     * that Baum Welsh trains generalize.
     * 
     * @param trainfile labeled data file to train HMM
     * @param evalfile unlabeled data file on which to do posterior decoding
     */
    static void BWtrainAndEval(String trainfile,String evalfile) throws Exception{
        String[] state = { "N", "E" };
        Distribution[] eprob = getEProbs();
 
        // train the model
        EmissionSequence[] es = 
        {EmissionSequence.readFromFile(trainfile)};
        HMM model = HMM.baumwelch(es,state,eprob,0.1);

        // evaluate the trained model on the evaluation data
        Report.printReport(model,EmissionSequence.readFromFile(evalfile),true,printLabels);
    }

    /** Simple test to verify that labeled training works. */
    static void labeledTest(){
        String[] state = { "N", "E" };
        Distribution[] eprob = getOldEProbs();
        double seq[][] = {{0.1,1.83},{0.001,5.2},{0.03,3.21}, // Ns
                        {3.2,1.3},{3.1,1.1},{3.3,1.1},{3.2,1.2}, // Es
                        {0.04,3.21},{0.02,1.1}, // Ns
                        {6.1,3.4},{6.4,3.2},{6.2,3.3}}; // Es
        int labels[] = {0,0,0,1,1,1,1,0,0,1,1,1};

        EmissionSequence[] es = { EmissionSequence.toVectorSeq(seq) };
        LabeledEmissionSequence[] les = {new LabeledEmissionSequence(es[0],labels)};
        HMM bw = HMM.baumwelch(es, state, eprob, 0.5);
        System.out.println("Baum Welsh");
        Report.printReport(bw,es[0],true,printLabels);
        System.out.println("Right solution:\nNNNEEEENNEEE");
        HMM labeled = HMM.labeledTrain(les, state, eprob);
        System.out.println("Labeled Training");
        Report.printReport(labeled,les[0],true,printLabels);
        System.out.println("Right solution:\nNNNEEEENNEEE");

    }

    //////////////// Functions below this point are not accessible through args
    // They tend to be messy testing functions.

    static void finalTest(){
        String[] state = { "N", "E" };
 
        //                       A ~ G(a,t)   phi ~ U(0,2pi)
        Distribution[] Ndists = {Gamma.rand(),new Uniform(0,2*Math.PI)};
        //Distribution[] Ndists = {Exp.rand(),new Uniform(0,2*Math.PI)};
        //                       A ~ G(a,t)
        Distribution[] Edists = {Gamma.rand(),
        //  phi_i - phi_{i-1} ~ N(0,d^2)
            new CenteredNorm(0,Math.random()*2+1,new DiffTransform())};
        //Distribution[] Edists = {
            //new CenteredNorm(0,Math.random()*2+1,new DiffTransform()),
            //new CenteredNorm(0,Math.random()*2+1,new DiffTransform())};
        Distribution[] eprob = { /*N*/ new VectorDist(Ndists),
                                 /*E*/ new VectorDist(Edists)};
        
        // Gamma has problems with 0??? - yes, there's a division by theta,
        // and gamma(alpha), which is divergent at 0.
        double seq[][] = {{0.1,1.83},{0.001,5.2},{0.03,3.21}, // Ns
                        {3.2,1.3},{3.1,1.1},{3.3,1.1},{3.2,1.2}, // Es
                        {0.04,3.21},{0.02,1.1}, // Ns
                        {6.1,3.4},{6.4,3.2},{6.2,3.3}}; // Es
        //double seq[] = {3,1,5,1,1,6,2,4,1,1,1,2,1,1,2,3,6,5,3,5,3,2,4,5};
        /*"315116246446644245311321631164152133625144543631656626566666"
          + "651166453132651245636664631636663162326455236266666625151631"
          + "222555441666566563564324364131513465146353411126414626253356"
          + "366163666466232534413661661163252562462255265252266435353336"
          + "233121625364414432335163243633665562466662632666612355245242";*/
        //hmm.print(new SystemOut());
        System.out.println("----------------------------------");
        System.out.println("Sequence:");
        for(int i=0;i<seq.length;i++){
            for(int j=0;j<seq[0].length;j++){
                System.out.print(seq[i][j]+" ");
            }
            System.out.println();
        }
        System.out.print("\n");

        EmissionSequence[] es = { EmissionSequence.toVectorSeq(seq) };
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.00001);
        Report.printReport(estimate,es[0]);
        System.out.println("Right solution:\nNNNEEEENNEEE");
    }

    static void gammaTest() {
        String[] state = { "S", "T" };
        Distribution[] eprob = { Gamma.rand(),Gamma.rand()};//{new Exp(1),new Exp(1)};

        double seq[] = {0.001,0.05,0.07,0.1,0.1,0.01,1,2,3,4,5,4,3,2,3};
        //double seq[] = {3,1,5,1,1,6,2,4,1,1,1,2,1,1,2,3,6,5,3,5,3,2,4,5};
        /*"315116246446644245311321631164152133625144543631656626566666"
          + "651166453132651245636664631636663162326455236266666625151631"
          + "222555441666566563564324364131513465146353411126414626253356"
          + "366163666466232534413661661163252562462255265252266435353336"
          + "233121625364414432335163243633665562466662632666612355245242";*/
        //hmm.print(new SystemOut());
        System.out.println("----------------------------------");
        System.out.println("Sequence:");
        for(int i=0;i<seq.length;i++){
            System.out.print(seq[i]+" ");
        }
        System.out.print("\n");

        EmissionSequence[] es = { EmissionSequence.toRealSeq(seq) };
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.00001);
        Report.printReport(estimate,es[0]);
    }

    static void diffTest() {
        String[] state = { "N", "E" };
        //Distribution E = Norm.rand();//new CenteredNorm(0,1);//
        Distribution[] eprob = { new Uniform(1,10),Norm.rand(new DiffTransform())};

        //double seq[] = {0.001,0.05,0.07,0.1,0.1,0.01,1,2,3,4,5,4,3,2,3};
        //double seq[] = {1,10,2,11,2,10,1,10.5,11,1.2,0.9,10,1,11};
        // simulated sequence of phase values (values 1-10)
        double seq[] = {3,9,1,1,3,4,4,3,4,5,4,8,5,3,7,9,9,8,9,9,9,8,4,1,1,6,2,4,9,9,4,2,6};
        System.out.println("----------------------------------");
        System.out.println("Sequence:");
        for(int i=0;i<seq.length;i++){
            System.out.print(seq[i]+" ");
        }
        System.out.print("\n");

        EmissionSequence[] es = { EmissionSequence.toRealSeq(seq) };
        //es[0].assignFilter(E,EmissionSequence.calcDifferences(seq));
        //es[0].printSequenceFor(E);
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.00001);
        Report.printReport(estimate,es[0]);
        //estimate.print(new SystemOut());
        System.out.println("Best solution:");
        Viterbi vit = new Viterbi(estimate, es[0]);
        System.out.println(vit.getPath());
    }

    static void normTest() {
        String[] state = { "S", "T" };
        Distribution[] eprob = { Norm.rand(),Norm.rand()};

        //double seq[] = {0.001,0.05,0.07,0.1,0.1,0.01,1,2,3,4,5,4,3,2,3};
        double seq[] = {1,10,2,11,2,10,1,10.5,11,1.2,0.9,10,1,11};
        System.out.println("----------------------------------");
        System.out.println("Sequence:");
        for(int i=0;i<seq.length;i++){
            System.out.print(seq[i]+" ");
        }
        System.out.print("\n");

        EmissionSequence[] es = { EmissionSequence.toRealSeq(seq) };
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.00001);
        estimate.print(new SystemOut());
        System.out.println("Best solution:");
        Viterbi vit = new Viterbi(estimate, EmissionSequence.toRealSeq(seq));
        System.out.println(vit.getPath());
    }

    static void expTest() {
        String[] state = { "S", "T" };
        Distribution[] eprob = { Exp.rand(),Exp.rand()};//{new Exp(1),new Exp(1)};

        double seq[] = {0.001,0.05,0.07,0.1,0.1,0.01,1,2,3,4,5,4,3,2,3};
        //double seq[] = {3,1,5,1,1,6,2,4,1,1,1,2,1,1,2,3,6,5,3,5,3,2,4,5};
        /*"315116246446644245311321631164152133625144543631656626566666"
          + "651166453132651245636664631636663162326455236266666625151631"
          + "222555441666566563564324364131513465146353411126414626253356"
          + "366163666466232534413661661163252562462255265252266435353336"
          + "233121625364414432335163243633665562466662632666612355245242";*/
        //hmm.print(new SystemOut());
        System.out.println("----------------------------------");
        System.out.println("Sequence:");
        for(int i=0;i<seq.length;i++){
            System.out.print(seq[i]+" ");
        }
        System.out.print("\n");

        /*Forward fwd = new Forward(hmm,seq);
          fwd.print(new SystemOut());
          Backward bwd = new Backward(hmm, seq);
          bwd.print(new SystemOut());*/
        //Viterbi vit = new Viterbi(hmm, seq);
        //vit.print(new SystemOut());
        //System.out.println(vit.getPath());

        //      Viterbi vit = new Viterbi(hmm, x);
        //      // vit.print(new SystemOut());
        //      System.out.println(vit.getPath());
        //      Forward fwd = new Forward(hmm, x);
        //      //   fwd.print(new SystemOut());
        //      System.out.println(fwd.logprob());
        //      Backward bwd = new Backward(hmm, x);
        //      //    bwd.print(new SystemOut());
        //      System.out.println(bwd.logprob());
        //      PosteriorProb postp = new PosteriorProb(fwd, bwd);
        //      for (int i=0; i<x.length(); i++)
        //        System.out.println(postp.posterior(i, 1));
        EmissionSequence[] es = { EmissionSequence.toRealSeq(seq) };
        HMM estimate = HMM.baumwelch(es, state, eprob, 0.00001);
        estimate.print(new SystemOut());
        System.out.println("Best solution:");
        Viterbi vit = new Viterbi(estimate, EmissionSequence.toRealSeq(seq));
        System.out.println(vit.getPath());
    }

    /*static void CpG() {
      String[] state = { "A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-" };
      double p2m = 0.05;          // P(switch from plus to minus)
      double m2p = 0.01;          // P(switch from minus to plus)
      double[][] aprob = { 
      { 0.180-p2m, 0.274-p2m, 0.426-p2m, 0.120-p2m, p2m, p2m, p2m, p2m },
      { 0.171-p2m, 0.368-p2m, 0.274-p2m, 0.188-p2m, p2m, p2m, p2m, p2m },
      { 0.161-p2m, 0.339-p2m, 0.375-p2m, 0.125-p2m, p2m, p2m, p2m, p2m }, 
      { 0.079-p2m, 0.335-p2m, 0.384-p2m, 0.182-p2m, p2m, p2m, p2m, p2m },
      { m2p, m2p, m2p, m2p,  0.300-m2p, 0.205-m2p, 0.285-m2p, 0.210-m2p },
      { m2p, m2p, m2p, m2p,  0.322-m2p, 0.298-m2p, 0.078-m2p, 0.302-m2p },
      { m2p, m2p, m2p, m2p,  0.248-m2p, 0.246-m2p, 0.298-m2p, 0.208-m2p },
      { m2p, m2p, m2p, m2p,  0.177-m2p, 0.239-m2p, 0.292-m2p, 0.292-m2p } };

      String esym = "ACGT";
      double[][] eprob = { { 1, 0, 0, 0 },
      { 0, 1, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 1 },
      { 1, 0, 0, 0 },
      { 0, 1, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 1 } };

      HMM hmm = new HMM(state, aprob, esym, eprob);

      String x = "CGCG";
      Viterbi vit = new Viterbi(hmm, x);
      vit.print(new SystemOut());
      System.out.println(vit.getPath());
      }*/
}

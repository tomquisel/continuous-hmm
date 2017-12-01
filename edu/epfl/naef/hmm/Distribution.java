package edu.epfl.naef.hmm;

import java.util.*;

/** Abstract class describing what features a probability distribution
 * should implement. 
 */
abstract class Distribution{

    /** transform to perform on input values before calculating their
     * probabilities. A transformation is performed on an entire
     * EmissionSequence, not just individual Emissions in the sequence. This
     * is maximally general, and the speed hit is mitigated by caching results.
     */
    Transform transform;

    //Distribution(){}
    
    Distribution(Transform transform){
        this.transform=transform;
    }
    
    /** Calculate the probability of an emission in an EmissionSequence.
     *
     * @param xs EmissionSequence to consider
     * @param ind index of the event in xs to calculate the probability of
     * @return probability of xs.val()[ind] according to this distribution
     */
    abstract double pdf(EmissionSequence xs,int ind);
    /** same as pdf, but in log-space */
    abstract double lpdf(EmissionSequence xs,int ind);
    // used for determining the ML estimate of parameters
    // given an output sequence and the corresponding state table

    // estimates parameters using the ML (or preferably unbiased) estimator
    /** Given a sequence of emissions and the corresponding forward and
     * backward tables, calculates new distribution parameters using the ML
     * estimator.
     *
     * @param es array of emission sequences to use for parameter estimation
     * @param logP log-probability of the model over each EmissionSequence in es
     * @param ind index of the state whose parameters we are estimating
     * @param fwd forward log-probability table
     * @param bwd backward log-probability talbe
     */
    abstract void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd);

    //Distribution dup(){return null;}

    /** Return a distribution with random parameters. */
    static Distribution rand(){return rand(new IdentityTransform());}
    static Distribution rand(Transform t){return null;}

    // return the emission that this Distribution sees at time t and uses
    // to calculate its pdf
    /** Perform this Distribution's personal transform on an arbitrary
     * EmissionSequence. This is used for printing data to the user. Remember
     * that the transformation calculation is cached, so this is not as 
     * absurdly slow as it looks. It's amortized O(1), anyway.
     * 
     * @param es EmissionSequence to transform.
     * @param t specific emission in es whose transformed value we want to see
     * @return the transformed value of es.val()[t]
     */
    public Emission getTransformedInput(EmissionSequence es,int t){
        return transform.apply(es).val()[t];
    }

    // UTILITY FUNCTIONS
    // These are specific to certain types of distributions
    
    /** Unboxes a double from a RealEmission at position ind in es */
    protected double emissionAsDouble(EmissionSequence es,int ind){
        EmissionSequence after = transform.apply(es);
        return ((RealEmission)after.val()[ind]).val();
    }
    
    /** Calculates the log-probability, given the HMM, of being in state ind at
     * every time step. This operates on a set of emission sequences.
     *
     *@param logP the log-probability of each emission sequence given the model
     * @param ind the index of the state we're interested in
     * @param fwds the forward table for each emission sequence considered
     * @param bwds the backward table for each emission sequence considered
     * @return a two dimensional array of state log-probabilities
     */
    protected double[][] stateProbs(double[] logP,int ind, 
            Forward[] fwds, Backward[] bwds){

        double[][] prob = new double[fwds.length][];

        // loop through all the output sequences
        for(int s=0;s<fwds.length;s++){
            Forward fwd = fwds[s];
            prob[s] = new double[fwd.length()];
            for (int i=0; i<fwd.length(); i++) {
                // probability of being in state ind at time i given the 
                // observed sequence <=> p(pi[i][ind]|X)
                prob[s][i] = Math.exp(fwd.f[i+1][ind+1] 
                        + bwds[s].b[i+1][ind+1] 
                        - logP[s]);
            }
        }

        return prob;
    }

    /** Calculates the sample mean for the real-valued emission sequences es.
     * @param es array of real-valued emission sequences to take the mean of
     * @param prob the probability of being in a certain state for every point
     * @return the sample mean of es
     */
    protected double sampleMean(EmissionSequence[] es, double[][] prob){
        double mean = 0.0;
        double weight = 0.0;

        // loop through all the output sequences
        for(int s=0;s<es.length;s++){
            for (int i=0; i<es[0].length; i++) {
                //weight each output value by the probability that it was 
                //emitted by the state ind
                mean += prob[s][i] * emissionAsDouble(es[s],i);
                //sum up the total weight used so we can normalize the mean
                weight += prob[s][i];
            }
        }
        
        return mean/weight;
    }
    
    /** Calculates the sample variance for the real-valued emission sequences es.
     * @param es array of real-valued emission sequences to take the variance of
     * @param prob the probability of being in a certain state for every point
     * @param mean the sample mean for es
     * @return the sample variance of es
     */
    protected double sampleVariance(EmissionSequence[] es, double[][] prob, double mean){
        double var = 0.0;
        double weight = 0.0;
        
        // estimate the variance 
        // there is a lot of duplicated computation here. Going for code
        // clarity over speed for the moment.
        for(int s=0;s<es.length;s++){
            for(int i=0;i<es[0].length;i++){
                // distance of the sample from the sample mean
                double dist = emissionAsDouble(es[s],i) - mean;
                // add the weighted distance squared to the variance sum
                var += prob[s][i] * dist*dist;
                weight += prob[s][i];
            }
        }
        // normalize the variance
        return var/weight;
    
    }
    
}

/** Designed to handle states where a vector of emissions is modeled by an
 *  arbitrary vector of real-valued distributions. EXPECTS Emissions of type 
 *  VectorEmission.
 */
class VectorDist extends Distribution{
    /** vector of real-valued distributions */
    Distribution[] dist;
    /** vector of weights dictating how much of an impact each distribution has
     */
    double[] weight;

    /** allows speedup of the transformation performed by toSequenceArray */
    HashMap<EmissionSequence,EmissionSequence[]> cache = new HashMap();

    VectorDist(Distribution[] dist){
        this(dist,null,new IdentityTransform());
    }

    VectorDist(Distribution[] dist,double[] weight){
        this(dist,weight,new IdentityTransform());
    }

    VectorDist(Distribution[] dist,double[] weight, Transform t){
        super(t);
        this.dist = dist;
        if(weight == null){
            weight = new double[dist.length];
            for(int i=0;i<dist.length;i++)
                weight[i] = 0.5;
        }
        if(weight.length != dist.length){
            System.out.println("in VectorDist(): dist.length="+dist.length+
                    " weight.length="+weight.length+
                    ". weight & dist must be of equal lengths");
            System.exit(0);
        }
        this.weight = weight;
    }

    /** The pdf of a VectorDist is calculated by multiplying the probabilities
     * of the vector's components together. This makes an independence
     * assumption which is not necessarily justified, but seems to work well.
     */
    public double pdf(EmissionSequence es,int ind){
        double prob = 1.0;
        EmissionSequence[] ess = toSequenceArray(es);
        for(int i=0;i<dist.length;i++){
            prob *= Math.pow(dist[i].pdf(ess[i],ind),weight[i]);
        }
        return prob;
    }

    public double lpdf(EmissionSequence es,int ind){
        double prob = 0.0;
        EmissionSequence[] ess = toSequenceArray(es);
        for(int i=0;i<dist.length;i++){
                prob += weight[i]*dist[i].lpdf(ess[i],ind);
        }
        //System.out.println(this+"- lpdf(es["+es.val(null)[ind]+"]):"+prob);
        return prob;
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){

        for(int i=0;i<dist.length;i++){
            // scan the es array and extract the sequences for distribution i
            EmissionSequence[] ess = new EmissionSequence[es.length];
            for(int j=0;j<es.length;j++){
                ess[j] = toSequenceArray(es[j])[i];
            }
            dist[i].estimateParams(ess,logP,ind,fwd,bwd);
        }
        
    }

    // dup is useless, I think
    Distribution dup(){
        return null;
    }

    // this function doesn't really make sense for a VectorDist
    static Distribution rand(){
        return null;
    }

    public Emission getTransformedInput(EmissionSequence es,int t){
        // construct a vector of the values observed by each of the
        // member distributions
        
        double[] val = new double[dist.length];
        EmissionSequence[] ess = toSequenceArray(es);
        
        for(int i=0;i<dist.length;i++){
            val[i] = ((RealEmission)dist[i].
                    getTransformedInput(ess[i],t)).val();
        }
        return new VectorEmission(val);
    }

    public String toString(){
        String ret="VectorDist(";

        if(dist.length == 0)
            return ret+")";
        else
            ret += dist[0].toString();
        
        for(int i=1;i<dist.length;i++)
            ret += ","+dist[i].toString();
                
        return ret+")";
    }

    /** A hackish accessor function to dump Distribution parameters to file.*/
    public double[] getFirstGammaParams(){
        Gamma firstGamma = (Gamma) dist[0];
        return firstGamma.getParams();
    }

    /** Convert an EmissionSequence of VectorEmissions to an array of
     * EmissionSequences of RealEmissions. This function being efficient is
     * critical to a VectorDistribution, since any time a pdf is asked
     * for, or parameters are estimated, this conversion must be performed.
     * @param es EmissionSequence of VectorEmissions
     * @return array of EmissionSequences of RealEmissions
     */
    protected EmissionSequence[] toSequenceArray(EmissionSequence es){

        // apply the transform
        EmissionSequence after = transform.apply(es);
        
        // check the cache to see if we can skip recomputation
        EmissionSequence[] ess = cache.get(after);
        if(ess != null)
            return ess;
        
        ess = new EmissionSequence[dist.length];
        for(int i=0;i<dist.length;i++){
            RealEmission[] re = new RealEmission[after.length];
            for(int j=0;j<re.length;j++){
                re[j] = new RealEmission(
                        ((VectorEmission)(after.val()[j])).val()[i]);
            }
            ess[i] = new EmissionSequence(re);
        }

        // put the result into the cache
        cache.put(after,ess);

        return ess;
    }
}

/** Implements the Gamma Distribution. EXPECTS Emissions of type RealEmission*/
class Gamma extends Distribution{
    double alpha;
    double theta;
    double logNormFact;

    Gamma(double alpha,double theta){
        this(alpha,theta,new IdentityTransform());
    }

    Gamma(double alpha, double theta, Transform transform){
        super(transform);

        this.alpha=alpha; 
        this.theta=theta;
        // normalization factor ( 1/(gamma(alpha)*theta^alpha) )
        logNormFact = org.apache.commons.math.special.Gamma.logGamma(alpha)
            + alpha * Math.log(theta);
    }

    public double pdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        return Math.pow(x,alpha-1) * Math.exp(-x/theta) / Math.exp(logNormFact);
    }

    public double lpdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        return (alpha-1)*Math.log(x) - x/theta - logNormFact;
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){
        
        // probability of being in state ind at time i in sequence s
        double[][] probs = stateProbs(logP,ind,fwd,bwd);
        double mean = sampleMean(es,probs);
        double var = sampleVariance(es,probs,mean);
        
        // the ML estimate of theta is var/mean
        theta = var/mean;
        // the ML estimate of alpha is mean/theta
        alpha = mean/theta;

        // avoid degenerate solutions
        if(alpha > 300){
            alpha = 300;
        }
        if(theta < 0.001){
            theta = 0.001;
        }
        
        // calculate values used to speed up lpdf computation
        logNormFact = org.apache.commons.math.special.Gamma.logGamma(alpha)
            + alpha * Math.log(theta);
    }

    Distribution dup(){
        return new Gamma(alpha,theta);
    }

    static Distribution rand(){
        return rand(new IdentityTransform());
    }

    static Distribution rand(Transform t){
        // random alpha and theta between 1 and 6
        return new Gamma(Math.random()*5+1,Math.random()*5+1,t);
    }

    public String toString(){
        return "Gamma("+alpha+","+theta+")";
    }

    public double[] getParams(){
        double[] params = new double[2];
        params[0] = alpha;
        params[1] = theta;
        return params;
    }
}

/** Implements the Exponential Distribution. EXPECTS Emissions of type
 * RealEmission
 */
class Exp extends Distribution{
    double lambda;
    double logLambda;

    Exp(double lambda){
        this(lambda,new IdentityTransform());
    }

    Exp(double lambda, Transform transform){
        super(transform);
        
        this.lambda=lambda; 
        logLambda = Math.log(lambda);
    }

    public double pdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        return lambda*Math.exp(-lambda*x);
    }

    public double lpdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        return logLambda-lambda*x;
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){
        
        // probability of being in state ind at time i in sequence s
        double[][] probs = stateProbs(logP,ind,fwd,bwd);
        double mean = sampleMean(es,probs);
        
        // the ML estimate of lambda is 1/u where u is the sample mean
        lambda = 1.0/mean;
        // calculate values used to speed up lpdf computation
        logLambda = Math.log(lambda);
    }

    Distribution dup(){
        return new Exp(lambda);
    }

    static Distribution rand(){
        return rand(new IdentityTransform());
    }

    static Distribution rand(Transform t){
        // random lambda between 0 & 1 (this is not the entire domain)
        return new Exp(Math.random(),t);
    }

    public String toString(){
        return "Exp("+lambda+")";
    }
}

/**Implements the Normal Distribution. EXPECTS Emissions of type RealEmission*/
class Norm extends Distribution{
    double mean;
    double var;
    double normFact;
    
    Norm(double mean,double var){
        this(mean,var,new IdentityTransform());
    }

    Norm(double mean, double var, Transform transform){
        super(transform);

        this.mean = mean;
        this.var = var;
        normFact = Math.sqrt(var*2.0*Math.PI);
    }
    
    public double pdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        double diff = x-mean;
        return HMM.exp(-(diff*diff)/(2.0*var)) / normFact;
    }
    
    public double lpdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        double diff = x-mean;
        return -(diff*diff)/(2.0*var) - Math.log(normFact);
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){
        
        // probability of being in state ind at time i in sequence s
        double[][] probs = stateProbs(logP,ind,fwd,bwd);
        mean = sampleMean(es,probs);
        // the unbiased estimator of variance is actually n/(n-1)*var, but it 
        // is unclear how to calculate n-1 when each sample is not equally 
        // weighted. We will have to go with a slightly biased estimator.
        var = sampleVariance(es,probs,mean);

        // used to prevent certain degenerate "optimal" solutions where
        // a Norm(x,0) results, with an infinite pdf value at x.
        // There are almost never interesting solutions.
        /*if(var<0.3){
            var = 0.3;
        }*/

        // calculate values used to speed up lpdf computation
        normFact = Math.sqrt(var*2.0*Math.PI);
        
    }

    Distribution dup(){
        return new Norm(mean,var);
    }

    static Distribution rand(){
        return rand(new IdentityTransform());
    }

    static Distribution rand(Transform t){
        // random mean from -2 to 2, and variance from 0 to 10 (this is not the entire domain)
        return new Norm(Math.random()*4.0-2,Math.random()*10.0,t);
    }

    public String toString(){
        return "Norm("+mean+","+var+")";
    }
}

/** just like Norm, except parameter estimate will not change the mean.
 * useful for situations where you already know what you want the mean to be.
 */
class CenteredNorm extends Norm{
    
    CenteredNorm(double mean, double var){
        super(mean,var);
    }
    
    CenteredNorm(double mean, double var, Transform transform){
        super(mean,var,transform);
    }

    // estimates parameters using the ML (or preferably unbiased) estimator
    // DOES NOT adjust the value of mean
    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){
        
        // probability of being in state ind at time i in sequence s
        double[][] probs = stateProbs(logP,ind,fwd,bwd);
        var = sampleVariance(es,probs,mean);

        // used to prevent certain degenerate "optimal" solutions where
        // a Norm(x,0) results, with an infinite pdf value at x.
        // There are almost never interesting solutions.
        if(var<0.002){
            var = 0.002;
        }

        // calculate values used to speed up lpdf computation
        normFact = Math.sqrt(var*2.0*Math.PI);
    }

    Distribution dup(){
        return new CenteredNorm(mean,var);
    }

    public String toString(){
        return "CenteredNorm("+mean+","+var+")";
    }
}

/** Implements the Uniform Distribution. EXPECTS emissions of type 
 * RealEmission. Note that this is the uniform distribution on a
 * closed interval. If you create new Uniform(0,1), then the pdf
 * at 0 and 1 will be 1, not 0.
 */
class Uniform extends Distribution{
    double low;
    double high;
    double val,logVal;

    Uniform(double low,double high){
        this(low,high,new IdentityTransform());
    }
    
    Uniform(double low,double high,Transform transform){
        super(transform);
        
        this.low = low;
        this.high = high;
        val = 1.0 / (high - low);
        logVal = Math.log(val);
    }
    
    public double pdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        if(x>=low && x <= high)
            return val;
        return 0.0;
    }
    
    public double lpdf(EmissionSequence es,int ind){
        double x = emissionAsDouble(es,ind);
        if(x>=low && x <= high)
            return logVal;
        return Double.NEGATIVE_INFINITY;
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){}

    static Distribution rand(){
        return rand(new IdentityTransform());
    }

    static Distribution rand(Transform t){
        double low = Math.random()*3;
        return new Uniform(low, low + Math.random()*10,t);
    }

    Distribution dup(){
        return new Uniform(low,high);
    }

    public String toString(){
        return "Uniform("+low+","+high+")";
    }
}

/** Empty distribution. Useful as a default */
class Null extends Distribution{

    Null(){this(new IdentityTransform());}
    Null(Transform transform){super(transform);}
    
    public double pdf(EmissionSequence es,int ind){
        return 0.0;
    }
    public double lpdf(EmissionSequence es,int ind){
        return Double.NEGATIVE_INFINITY;
    }

    void estimateParams(EmissionSequence[] es, double[] logP,int ind,
            Forward[] fwd, Backward[] bwd){}

    Distribution dup(){
        return new Null();
    }

    static Distribution rand(){
        return new Null();
    }

    static Distribution rand(Transform t){
        return new Null();
    }

    public String toString(){
        return "Null()";
    }
}


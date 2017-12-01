package edu.epfl.naef.hmm;

import java.util.*;

/** An abstract class representing a general function 
 * f:EmissionSequence-&gt;EmissionSequence. This is useful for cases where you
 * want to transform your emission values before giving them to the HMM. For
 * example, if you want to have your HMM model the first order difference
 * of your data, instead of the data itself, this class is very useful.*/
abstract class Transform{
    // this allows nesting transforms ( DiffTransform(MultTransform()) )
    Transform next;
    // allows caching results 
    // this could cause high memory usage if used improperly, but is necessary
    // to avoid recomputing the transform for the entire sequence every time
    // a pdf is calculated.
    HashMap cache = new HashMap();
    
    Transform(){
        next = null;
    }
    
    Transform(Transform next){
        this.next = next;
    }

    /** apply the next Transform in the transform chain. */
    protected EmissionSequence applyNext(EmissionSequence es){
        if(next == null)
            return es;
        else
            return next.apply(es);
    }

    /** check the cache to see if we've already transformed es once */
    protected EmissionSequence getResult(EmissionSequence es){
        return (EmissionSequence)cache.get(es);
    }

    /** put a transformation result into the cache */
    protected void putResult(EmissionSequence in,EmissionSequence out){
        cache.put(in,out);
    }

    /** Handles caching of results and applying the next transform.
     * This is wrapper for applyTransform to make inheriting Transform easier.
     * Now, you don't have to worry about caching to implement a transform,
     * it happens automagically. Yay!
     */
    EmissionSequence apply(EmissionSequence es){
        EmissionSequence after = applyNext(es);
        // try the cache
        EmissionSequence res = getResult(after);
        if(res != null)
            return res;

        // call the specific transform
        res = applyTransform(after);

        // add result to the cache
        putResult(after,res);
        
        return res;
    }

    abstract protected EmissionSequence applyTransform(EmissionSequence es);
}

/** The identity function. */
class IdentityTransform extends Transform{
    // Apply is overriden instead of applyTransform because there is no need
    // for caching
    EmissionSequence apply(EmissionSequence es){ return es; }
    protected EmissionSequence applyTransform(EmissionSequence es)
    {return null; }
}

/** Calculates the first order difference of an EmissionSequence.
 * Expects a sequence of RealEmissions. */
class DiffTransform extends Transform{
    protected EmissionSequence applyTransform(EmissionSequence es){
        // only works for RealEmission sequences
        RealEmission[] xs = (RealEmission[])es.val();
        
        RealEmission[] diffs = new RealEmission[xs.length];
        diffs[0] = new RealEmission(0.0);
        for(int i=1;i<xs.length;i++)
            diffs[i] = new RealEmission(xs[i].val()-xs[i-1].val());

        return new EmissionSequence(diffs);
    }
}

/** computes a circular distance between each adjacent pair of emissions.
 * circular distance is useful for arguments of modular functions
 * for CircDistTransform(24), circdist(a,b) = min(abs(a-b),24-abs(a-b))
 *  Expects RealEmissions
 */
class CircDistTransform extends Transform{
    double max;

    // assumes values will be between 0 and max
    CircDistTransform(double max){
        this(max,null);
    }
    CircDistTransform(double max,Transform next){
        super(next);
        this.max = max;
    }
    
    protected EmissionSequence applyTransform(EmissionSequence es){
        // only works for RealEmission sequences
        RealEmission[] xs = (RealEmission[])es.val();
        
        RealEmission[] dists = new RealEmission[xs.length];
        dists[0] = new RealEmission(0.0);
        for(int i=1;i<xs.length;i++){
            double dist = Math.abs(xs[i].val()-xs[i-1].val());
            if(dist <= max-dist){
                if(xs[i].val() < xs[i-1].val())
                    dist = - dist;
            }
            else{
                dist = max-dist;
                if(xs[i].val() > xs[i-1].val())
                    dist = - dist;
            }
            dists[i] = new RealEmission(dist);
        }
        return new EmissionSequence(dists);
    }
}


/** Computes the negative log of each emission.
 *  Expects RealEmissions
 *  */
class NegLogTransform extends Transform{

    NegLogTransform(){
        super();
    }

    NegLogTransform(Transform next){
        super(next);
    }

    protected EmissionSequence applyTransform(EmissionSequence es){
        // only works for RealEmission sequences
        RealEmission[] xs = (RealEmission[])es.val();
        
        RealEmission[] neglog = new RealEmission[xs.length];
        for(int i=0;i<xs.length;i++){
            double x = xs[i].val();
            neglog[i] = new RealEmission(-Math.log(x));
        }
        return new EmissionSequence(neglog);
    }
}

/** calculates 1-e for each RealEmission e.
 * Expects RealEmissions.
 */
class OneMinusTransform extends Transform{

    OneMinusTransform(){
        super();
    }
    OneMinusTransform(Transform next){
        super(next);
    }

    protected EmissionSequence applyTransform(EmissionSequence es){
        // only works for RealEmission sequences
        RealEmission[] xs = (RealEmission[])es.val();
        
        RealEmission[] oneminus = new RealEmission[xs.length];
        for(int i=0;i<xs.length;i++){
            double x = xs[i].val();
            oneminus[i] = new RealEmission(1-x);
        }
        return new EmissionSequence(oneminus);
    }
}

/** Ensures that no Emissions are exactly 0, but just small positive reals.
 * This is generally used with the gamma distribution because it does not
 * handle 0s well 
 * Expects RealEmissions.
 */
class ZeroGuard extends Transform{
    double min;

    // min must be positive, the input sequence is assumed to be non-negative
    // if there are negative vales in the sequence, their absolute value is taken
    ZeroGuard(double min){
        this(min,null);
    }
    ZeroGuard(double min,Transform next){
        super(next);
        this.min = min;
    }
    
    protected EmissionSequence applyTransform(EmissionSequence es){
        // only works for RealEmission sequences
        RealEmission[] xs = (RealEmission[])es.val();
        
        RealEmission[] guarded = new RealEmission[xs.length];
        for(int i=0;i<xs.length;i++){
            double x = xs[i].val();
            if(x<0){
                x = -x;
            }
            if(x < min){
                x = min;
            }
            guarded[i] = new RealEmission(x);
        }
        return new EmissionSequence(guarded);
    }
}


(ns ^{:doc
      "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/Distribution
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/UniformDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/NormalDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/BinomialDistribution.html"}
  dl4clj.nn.conf.distribution.distribution
  (:import [org.deeplearning4j.nn.conf.distribution
            Distribution UniformDistribution NormalDistribution BinomialDistribution]))

(defmulti distribution (fn [opts] (first (keys opts))))

(defn uniform-distribution [lower upper]
  (UniformDistribution. lower upper))

(defn normal-distribution [mean std]
  (NormalDistribution. mean std))

(defn binomial-distribution [number-of-trials probability-of-success]
  (BinomialDistribution. number-of-trials probability-of-success))

(defmethod distribution :uniform [opt]
  (uniform-distribution (:lower (:uniform opt)) (:upper (:uniform opt))))

(defmethod distribution :normal [opt]
  (normal-distribution (:mean (:normal opt)) (:std (:normal opt))))

(defmethod distribution :binomial [opt]
  (binomial-distribution (:number-of-trials (:binomial opt)) (:probability-of-success (:binomial opt))))

(comment

  (uniform-distribution 0.3 10)
  (distribution {:uniform {:lower -0.01, :upper 0.01}})

  (normal-distribution 10 0.3)
  (distribution {:normal {:mean 0.0, :std 0.3}})

  (binomial-distribution 10 0.3)
  (distribution {:binomial {:number-of-trials 0, :probability-of-success 0.08}})
  )

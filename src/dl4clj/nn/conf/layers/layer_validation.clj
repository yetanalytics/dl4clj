(ns dl4clj.nn.conf.layers.layer-validation
  (:import [org.deeplearning4j.nn.conf.layers LayerValidation])
  (:require [dl4clj.nn.conf.distributions :as distribution]))

(defn general-validation!
  "validates a layer params are reasonable

  :layer-name (str), the name of the layer

  :layer (layer), the nn layer

  :use-regularization? (boolean), is the layer set to use regularization?

  :use-drop-connect? (boolean), is the layer set to use drop connect

  :drop-out (double), the value used if use-drop-connect? is true

  :l2 (double), the l2 coef

  :l2-bias (double), the l2 coef for the bias

  :l1 (double), the l1 coef

  :l1-bias (double), the l1 coef for the bias

  :dist (distribution), the distribution used to sample weights from
   - you should not have to create a new distribution, just pass the one already
     used during layer setup

  returns nil"
  [& {:keys [layer-name layer use-regularization? use-drop-connect? drop-out
             l2 l2-bias l1 l1-bias dist]}]
  (LayerValidation/generalValidation layer-name layer use-regularization?
                                     use-drop-connect? drop-out l2 l2-bias
                                     l1 l1-bias (if map?
                                                  (distribution/distribution dist)
                                                  dist)))

(defn updater-validation!
  "Validate the updater configuration - setting the default updater values, if necessary

  :layer-name (str), the name of the layer

  :layer (layer), the layer itself

  :momentum (double), the momentum set for backprop

  :momentum-schedule (map), the schedule set during layer setup
   {iteration (int), momentum (double)}

  :adam-mean-decay (double), the adam mean decay value set at layer setup

  :adam-var-decay (double), the adam var decay value set at layer setup

  :rho (double), the rho value set at layer setup

  :rms-decay (double), the rms-decay value set at layer setup

  :epsilon (double), the epsilon value set at layer setup"
  [& {:keys [layer-name layer momentum momentum-schedule adam-mean-decay
             adam-var-decay rho rms-decay epsilon]}]
  (LayerValidation/updaterValidation layer-name layer momentum momentum-schedule
                      adam-mean-decay adam-var-decay rho rms-decay
                      epsilon))

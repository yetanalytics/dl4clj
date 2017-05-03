(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/plot/BarnesHutTsne.html"}
  dl4clj.plot.barnes-hut-tsne
  #_(:import [org.deeplearning4j.plot BarnesHutTsne BarnesHutTsne$Builder]
           [org.nd4j.linalg.api.ndarray INDArray]))

#_(defn builder [{:keys [invert-distance-metric        ;; (boolean)
                       learning-rate                 ;; (double)
                       min-gain                      ;; (double)
                       normalize                     ;; (boolean)
                       perplexity                    ;; (double)
                       final-momentum            ;; (double)
                       initial-momentum          ;; (double)
                       max-iter                  ;; (int)
                       momentum                  ;; (double)
                       real-min                  ;; (double)
                       switch-momentum-iteration ;; (int)
                       similarity-function           ;; (String)
                       stop-lying-iteration          ;; (int)
                       theta                         ;; (double)
                       tolerance                     ;; (double)
                       use-ada-grad                  ;; (boolean)
                       use-pca                       ;; (boolean)
                       ]
                :or {}
                :as opts}]
  (let [b (BarnesHutTsne$Builder.)]
    (when (or invert-distance-metric (contains? opts :invert-distance-metric))
      (.invertDistanceMetric b (boolean invert-distance-metric)))
    (when (or learning-rate (contains? opts :learning-rate))
      (.learningRate b (double learning-rate)))
    (when (or min-gain (contains? opts :min-gain))
      (.minGain b (double min-gain)))
    (when (or normalize (contains? opts :normalize))
      (.normalize b (boolean normalize)))
    (when (or perplexity (contains? opts :perplexity))
      (.perplexity b (double perplexity)))
    (when (or final-momentum (contains? opts :final-momentum))
      (.setFinalMomentum b (double final-momentum)))
    (when (or initial-momentum (contains? opts :initial-momentum))
      (.setInitialMomentum b (double initial-momentum)))
    (when (or max-iter (contains? opts :max-iter))
      (.setMaxIter b (int max-iter)))
    (when (or momentum (contains? opts :momentum))
      (.setMomentum b (double momentum)))
    (when (or real-min (contains? opts :real-min))
      (.setRealMin b (double real-min)))
    (when (or switch-momentum-iteration (contains? opts :switch-momentum-iteration))
      (.setSwitchMomentumIteration b (int switch-momentum-iteration)))
    (when (or similarity-function (contains? opts :similarity-function))
      (.similarityFunction b (str similarity-function)))
    (when (or stop-lying-iteration (contains? opts :stop-lying-iteration))
      (.stopLyingIteration b (int stop-lying-iteration)))
    (when (or theta (contains? opts :theta))
      (.theta b (double theta)))
    (when (or tolerance (contains? opts :tolerance))
      (.tolerance b (double tolerance)))
    (when (or use-ada-grad (contains? opts :use-ada-grad))
      (.useAdaGrad b (boolean use-ada-grad)))
    (when (or use-pca (contains? opts :use-pca))
      (.usePca b (boolean use-pca)))
    ))


;; broken
#_(defn barnes-hut-tsne
  ([opts]
   (.build ^BarnesHutTsne$Builder (builder opts)))
  ([^INDArray x
    ^INDArray y
    num-dimensions
    perplexity
    theta
    max-iter
    stop-lying-iteration
    momentum-switch-iteration
    momentum
    final-momentum
    learning-rate]
   (BarnesHutTsne. x y (int num-dimensions) (double perplexity) (double theta)
                   (int max-iter) (int stop-lying-iteration) (int momentum-switch-iteration)
                   (double momentum) (double final-momentum) (double learning-rate)))
  ([^INDArray x
    ^INDArray y
    num-dimensions
    ^String simiarlity-function
    theta
    ^Boolean invert
    max-iter
    real-min
    initial-momentum
    final-momentum
    momentum
    switch-momentum-iteration
    ^Boolean normalize
    ^Boolean use-pca
    stop-lying-iteration
    tolerance
    learning-rate
    ^Boolean use-ada-grad
    perplexity
    min-gain]
   (BarnesHutTsne. x y (int num-dimensions)
                   (str simiarlity-function)
                   (double theta) invert
                   (int max-iter) (double real-min)
                   (double initial-momentum)
                   (double final-momentum)
                   (double momentum)
                   (int switch-momentum-iteration)
                   normalize use-pca (int stop-lying-iteration)
                   (double tolerance) (double learning-rate)
                   use-ada-grad (double perplexity) (double min-gain))))

(ns dl4clj.datasets.iterator.impl.default-datasets
  (:refer-clojure :exclude [reset!])
  (:import [org.deeplearning4j.datasets.iterator.impl
            CifarDataSetIterator IrisDataSetIterator LFWDataSetIterator
            MnistDataSetIterator RawMnistDataSetIterator])
  (:require [dl4clj.utils :refer [contains-many?]]
            [datavec.api.split :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prebuilt dataset iterators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-cifar-data-set-iterator
  "Load the images from the cifar dataset,

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/CifarDataSetIterator.html
  and: https://github.com/szagoruyko/cifar.torch

  :batch-size (int), the batch size
  :n-examples (int), the number of examples from the ds to include in the iterator
  :img-dim (int-array), desired dimensions of the images
  :train? (boolean), are we training or testing?
  :use-special-pre-process-cifar? (boolean), are we going to use the predefined preprocessor built for this dataset
   - There is a special preProcessor used to normalize the dataset based on Sergey Zagoruyko example https://github.com/szagoruyko/cifar.torch
  :img-transform (map) config map for an image-transformation (as of writing this doc string, not implemented)
  :n-possible-labels (int), specify the number of possible outputs/tags/classes for a given image"
  [& {:keys [batch-size n-examples img-dims train?
             use-special-pre-process-cifar?
             n-possible-labels img-transform]
      :as opts}]
  (cond (contains-many? opts :batch-size :n-examples :img-dims :n-possible-labels
                        :img-transform :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img-dims n-possible-labels
                               img-transform use-special-pre-process-cifar? train?)
        (contains-many? opts :batch-size :n-examples :img-dims :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img-dims use-special-pre-process-cifar? train?)
        (contains-many? opts :batch-size :n-examples :img-dims :train?)
        (CifarDataSetIterator. batch-size n-examples img-dims train?)
        (contains-many? opts :batch-size :n-examples :img-dims)
        (CifarDataSetIterator. batch-size n-examples img-dims)
        (contains-many? opts :batch-size :n-examples :train?)
        (CifarDataSetIterator. batch-size n-examples train?)
        (contains-many? :batch-size :img-dims)
        (CifarDataSetIterator. batch-size img-dims)
        (contains-many? :batch-size :n-examples)
        (CifarDataSetIterator. batch-size n-examples)
        :else
        (assert (and (contains? opts :batch-size)
                     (or (contains? opts :img-dims)
                         (contains? opts :n-examples)))
                "you must provide atleast a batch size and number of examples or a batch size and the desired demensions of the images")))

(defn new-iris-data-set-iterator
  "IrisDataSetIterator handles traversing through the Iris Data Set.

  :batch (int), size of the batch
  :n-examples (int), number of examples to iterator over

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.html"
  [& {:keys [batch n-examples]}]
  (IrisDataSetIterator. batch n-examples))

(defn new-lfw-data-set-iterator
  "Creates a dataset iterator for the LFW image dataset.

  :img-dims (int-array), desired dimensions of the images
  :batch-size (int), the batch size
  :n-examples (int), number of examples to take from the dataset
  :use-subset? (boolean), use a subset of the dataset or the whole thing
  :train? (boolean, are we training a net or testing it
  :split-train-test (double), the division between training and testing datasets
  :n-labels (int), the number of possible classifications for a single image
  :rng (java.util.Random), a random value for shuffling the dataset before sampling from it
   -random number to lock in batch shuffling
  :label-generator (label generator), call (new-parent-path-label-generator) or
   (new-pattern-path-label-generator opts) to create a label generator to use
  :image-transform (map), a transform to apply to the images,
   - as of writing this doc string, this functionality not implemented

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.html"
  [& {:keys [img-dims batch-size n-examples use-subset? train? split-train-test
             n-labels rng label-generator image-transform]
      :as opts}]
  (cond (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng :image-transform)
        (LFWDataSetIterator. batch-size n-examples img-dims n-labels use-subset?
                             label-generator train? split-train-test image-transform
                             rng)
        (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img-dims n-labels use-subset?
                             label-generator train? split-train-test rng)
        (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img-dims n-labels use-subset?
                            train? split-train-test rng)
        (contains-many? opts :batch-size :n-examples :n-labels :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples n-labels train? split-train-test)
        (contains-many? opts :batch-size :n-examples :img-dims :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples img-dims train? split-train-test)
        (contains-many? opts :batch-size :n-examples :img-dims)
        (LFWDataSetIterator. batch-size n-examples img-dims)
        (contains-many? opts :batch-size :img-dims :use-subset?)
        (LFWDataSetIterator. batch-size img-dims use-subset?)
        (contains-many? opts :batch-size :n-examples)
        (LFWDataSetIterator. batch-size n-examples)
        (contains? opts :img-dims)
        (LFWDataSetIterator. img-dims)
        :else
        (assert false "you must supply atleast the desired image dimensions for the data")))

(defn new-mnist-data-set-iterator
  "creates a dataset iterator for the Mnist dataset

  :batch-size (int), the batch size
  :train? (boolean), training or testing
  :seed (int), int used to randomize the dataset
  :n-examples (int), the overall number of examples
  :binarize? (boolean), whether to binarize mnist or not
  :shuffle? (boolean), whether to shuffle the dataset or not
  :rng-seed (long), random number generator seed to use when shuffling examples
  :batch (int), size of each patch

  - need to test to see if its necessary to seperate batch-size/batch

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.html"
  [& {:keys [batch-size train? seed n-examples binarize? shuffle? rng-seed batch]
      :as opts}]
  (cond (contains-many? opts :batch-size :train? :seed)
        (MnistDataSetIterator. batch-size train? seed)
        (contains-many? opts :batch :n-examples :binarize? :train? :shuffle? :rng-seed)
        (MnistDataSetIterator. batch n-examples binarize? train? shuffle? rng-seed)
        (contains-many? opts :batch :n-examples :binarize?)
        (MnistDataSetIterator. batch n-examples binarize?)
        (contains-many? opts :batch :n-examples)
        (MnistDataSetIterator. batch n-examples)
        :else
        (assert false "you must atleast supply a batch and number of examples")))

(defn new-raw-mnist-data-set-iterator
  "Mnist data with scaled pixels

  :batch (int) size of each patch
  :n-examples (int), the overall number of examples

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/RawMnistDataSetIterator.html"
  [& {:keys [batch n-examples]}]
  (RawMnistDataSetIterator. batch n-examples))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn async-supported?
  [iter]
  (.asyncSupported iter))

(defn get-labels
  [iter]
  (.getLabels iter))

(defn has-next?
  [iter]
  (.hasNext iter))

(defn reset-iter!
  [iter]
  (doto iter (.reset)))

(defn next-data-point
  [& {:keys [iter batch-size]
      :as opts}]
  (if (contains? opts :batch-size)
    (.next iter batch-size)
    (.next iter)))

(defn test!
  [& {:keys [iter n-examples batch-size]
      :as opts}]
  (cond (contains-many? opts :iter :n-examples :batch-size)
        (doto iter (.test n-examples batch-size))
        (contains-many? opts :iter :n-examples)
        (doto iter (.test n-examples))
        (contains? opts :iter)
        (doto iter (.test))
        :else
        (assert false "you must provide atleast an iterator")))

(defn total-examples
  [iter]
  (.totalExamples iter))

(defn train!
  [iter]
  (doto iter (.train)))

(defn set-pre-processor
  "sets a preprocessor for a dataset iterator.

  assumes that the pre-processor is selected/configured and then passed here"
  [& {:keys [iter pre-processor]}]
  (doto iter (.setPreProcessor pre-processor)))

(defn get-feature-matrix
  [data-from-iter]
  (.getFeatureMatrix data-from-iter))

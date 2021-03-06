/*
 * Copyright (c) 2018-2020
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-05-03     Jianjia Ma   The first version
 */

#ifndef __NNOM_DENSE_H__
#define __NNOM_DENSE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "nnom.h"
#include "nnom_layers.h"
#include "nnom_local.h"
#include "nnom_tensor.h"

typedef struct _nnom_dense_layer_t
{
	nnom_layer_t super;
	size_t output_unit;
	nnom_tensor_t *weight;
	nnom_tensor_t *bias;
} nnom_dense_layer_t;

// a machine interface for configuration
typedef struct _nnom_dense_config_t
{
	nnom_layer_config_t super;
	nnom_qtype_t qtype; 	//quantisation type(per channel or per layer)
	nnom_tensor_t *weight;
	nnom_tensor_t *bias;
	int8_t output_shift;   // not sure if we need that
} nnom_dense_config_t;

nnom_status_t dense_free(nnom_layer_t *layer);

#ifdef __cplusplus
}
#endif

#endif /* __NNOM_DENSE_H__ */

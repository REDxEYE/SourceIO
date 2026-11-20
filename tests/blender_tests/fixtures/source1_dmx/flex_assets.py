"""Build DMX combination-operator Text from Model Entry flex recipes."""

from __future__ import annotations

from SourceIO.blender_bindings.source1.model_export.prop_qc import ModelEntryRecipe, FlexControllerRecipe


def render_flex_controllers(entry: ModelEntryRecipe) -> str:
    from io_scene_valvesource import datamodel

    dm = datamodel.DataModel('model', 1)
    root = dm.add_element(f'flex_{entry.name}', id=f'{entry.name}/flex')
    operator = dm.add_element('combinationOperator', 'DmeCombinationOperator', id=f'{entry.name}/combination')
    root['combinationOperator'] = operator
    controls = operator['controls'] = datamodel.make_array([], datamodel.Element)
    recipes = entry.flex_rule_set.controls if entry.flex_rule_set is not None else entry.flex_controllers
    for index, recipe in enumerate(recipes):
        control = dm.add_element(recipe.name, 'DmeCombinationInputControl', id=f'{entry.name}/{index}/{recipe.name}')
        if isinstance(recipe, FlexControllerRecipe):
            control['rawControlNames'] = datamodel.make_array(recipe.raw_names, str)
        else:
            control['rawControlNames'] = datamodel.make_array([], str)
        control['stereo'] = recipe.stereo
        control['eyelid'] = recipe.eyelid
        control['flexMin'] = recipe.minimum
        control['flexMax'] = recipe.maximum
        control['wrinkleScales'] = datamodel.make_array(
            recipe.wrinkle_scales if isinstance(recipe, FlexControllerRecipe) else (), float)
        controls.append(control)
    values = [datamodel.Vector3([0.0, 0.0, 0.5]) for _ in controls]
    operator['controlValues'] = datamodel.make_array(values, datamodel.Vector3)
    operator['controlValuesLagged'] = datamodel.make_array(values, datamodel.Vector3)
    operator['usesLaggedValues'] = False
    operator['dominators'] = datamodel.make_array([], datamodel.Element)
    targets = operator['targets'] = datamodel.make_array([], datamodel.Element)
    if entry.flex_rule_set is not None:
        by_object = {}
        for rule in entry.flex_rule_set.rules:
            by_object.setdefault(rule.object_name, []).append(rule)
        for object_name, rules in by_object.items():
            target = dm.add_element(object_name, 'DmeFlexRules', id=f'{entry.name}/rules/{object_name}')
            target['deltaStates'] = datamodel.make_array([], datamodel.Element)
            target['deltaStateWeights'] = datamodel.make_array([], datamodel.Vector2)
            for index, rule in enumerate(rules):
                delta = dm.add_element(rule.shape_name, 'DmeFlexRuleExpression',
                                       id=f'{entry.name}/rules/{object_name}/{index}')
                delta['expr'] = rule.expression
                delta['result'] = 0.0
                target['deltaStates'].append(delta)
                target['deltaStateWeights'].append(datamodel.Vector2([0.0, 0.0]))
            targets.append(target)
    return dm.echo('keyvalues2', 1)

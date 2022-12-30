from ....library.utils.singleton import SingletonMeta
from . import vs
from .sfm import SFM, Dag, get_bone

sfm = SFM()


# noinspection PyPep8Naming
class SfmUtils(metaclass=SingletonMeta):

    @staticmethod
    def FindFirstDag(bone_names, required=False):
        """ Search for a dag node matching one of the names in the specified list, resturn when the
            first one is found. If required is true raise an error exception if no dag node is found"""
        sfm.pose_mode()
        for name in bone_names:
            dag = sfm.FindDag(name)
            if dag is not None:
                return dag
        if required:
            raise Exception(f'Unable to find dag of name:  {bone_names}')

    @staticmethod
    def GetDagFromNameOrObject(target, check_result=True):
        dag = target
        if isinstance(target, str):
            dag = sfm.FindDag(target)
        if check_result:
            assert dag is not None, "Paramter is not a valid dag node or name of a dag node"
        return dag

    @staticmethod
    def AddElementToRig(element, animSet):
        """ Adds the specified element to the current rig if there is a rig active """
        rig = sfm.GetCurrentRig()
        if rig is not None:
            rig.AddElement(element, animSet)
        return

    def CreateConstrainedHandle(self, handleName, target, bPosControl=True, bRotControl=True, bCreateControls=True):
        """ Method for creating a rig handle which is directly constrained to a specfified target"""

        if target is None:
            return None

        targetDag = self.GetDagFromNameOrObject(target)

        handle = sfm.CreateRigHandle(handleName, posControl=bPosControl, rotControl=bRotControl)
        sfm.edit_mode()
        handle_bone = get_bone(sfm.obj, handle, 'EDIT')
        target_bone = get_bone(sfm.obj, targetDag, 'EDIT')
        size = (handle_bone.tail - handle_bone.head).magnitude
        handle_bone.head = target_bone.head
        handle_bone.tail = target_bone.tail
        handle_bone.tail += (handle_bone.tail - handle_bone.head) * size * 2
        handle_bone.roll = target_bone.roll

        sfm.PushSelection()
        sfm.SelectDag(targetDag)
        sfm.SelectDag(handle)
        #
        sfm.PointConstraint(mo=False, controls=bCreateControls)
        sfm.OrientConstraint(mo=False, controls=bCreateControls)
        #
        sfm.PopSelection()

        return handle

    def CreateOffsetHandle(self, handleName, target, offset, bPosControl=True, bRotControl=True, refSpaceDag=None,
                           bCreateControls=True):
        """Method for creating a rig handle which is constrained to a specfified target with an offset"""

        targetDag = self.GetDagFromNameOrObject(target)

        # Save the current selection and make a new selection starting with the target dag
        sfm.PushSelection()
        sfm.SelectDag(targetDag)

        # Get the position and rotation of the target dag
        handlePos = sfm.GetPosition()
        handleRot = sfm.GetRotation()

        # Create the new rig handle
        handle = sfm.CreateRigHandle(handleName, pos=handlePos, rot=handleRot, posControl=bPosControl,
                                     rotControl=bRotControl)
        if refSpaceDag is None:
            sfm.Move(offset.x, offset.y, offset.z, handleName, space="World", relative=True)
        else:
            sfm.Move(offset.x, offset.y, offset.z, handleName, space="refObject", refObject=refSpaceDag, relative=True)

        # Add the new handle to the selection so that now
        # both the target and and handle are selected
        sfm.SelectDag(handle)

        # Parent constrain the handle ot the target
        # sfm.ParentConstraint(mo=True, controls=bCreateControls)

        # Restore the old selection
        sfm.PopSelection()

        return handle

    def Parent(self, child, parent, nLogMode=vs.REPARENT_LOGS_OVERWRITE, bMaintainWorldPos=True):
        """ Make the dag node specified as the child a child of the dag node specified as the parent. """

        if child is None:
            return

        childDag = self.GetDagFromNameOrObject(child)
        parentDag = self.GetDagFromNameOrObject(parent)

        sfm.ParentConstraint(parentDag, childDag, mo=bMaintainWorldPos)

        return

    def ParentMaintainWorld(self, child, parent):
        """ Parent one dag to another, and update the logs of the child such that the world space
        position and orientation are not changed """

        self.Parent(child, parent, vs.REPARENT_LOGS_MAINTAIN_WORLD, True)
        return

    def CreateChannel(self, name, type, defaultValue, animSet, shot):
        """ Create a channel and add it to the track group for the animation set """
        channel = vs.CreateElement("DmeChannel", name, shot.GetFileId())
        self.AddElementToRig(channel, animSet)

        # Create the log for the channel

        # Should be able to use CreateLog(), but there is a problem with the type conversion
        # of DmAttribute, so just create the log manually and assign it to the channel.
        # channel.CreateLog( vs.AT_FLOAT )
        # log = channel.log

        log = vs.CreateElement("DmeFloatLog", "float log", shot.GetFileId())
        channel.SetLog(log)
        log.SetKey(vs.DmeTime_t(0), defaultValue)

        # Add the channel to the channels clip of the animation set
        channelsClip = self.GetChannelsClipForAnimSet(animSet, shot)
        if channelsClip is not None:
            channelsClip.channels.AddToTail(channel)

        return channel

    def CreateControlAndChannel(self, name, type, value, animSet, shot):
        """Create a control and corresponding value channel """
        newControl = animSet.FindOrAddControl(name, False)
        if newControl is not None:
            self.AddElementToRig(newControl, animSet)
            self.AddAttributeToElement(newControl, "value", type, value)
            self.AddAttributeToElement(newControl, "defaultValue", type, value)
            channelName = name + "_channel"
            channel = self.CreateChannel(channelName, type, value, animSet, shot)
            if channel is not None:
                newControl.SetValue("channel", channel)
                channel.SetInput(newControl, "value")

        return newControl

    def CreateHandleAt(self, handleName, target, bPosControl=True, bRotControl=True):
        """ Create a rig handle with the position and orientation of the specified node """

        sfm.edit_mode()
        targetDag = self.GetDagFromNameOrObject(target)
        target = get_bone(sfm.obj, targetDag, 'EDIT')

        bone = sfm.obj.data.edit_bones.new(name=handleName)
        bone.head = target.head
        bone.tail = target.tail
        bone.roll = target.roll

        dag = Dag(bone)
        sfm.pose_mode()
        bone = get_bone(sfm.obj, dag)
        if not bPosControl:
            bone.lock_location = [True, True, True]
        if not bRotControl:
            bone.lock_rotation_w = True
            bone.lock_rotation[0] = [True, True, True]

        return dag

    def CreatePointOrientConstraint(self, target, slave, bCreateControls=True, group=None):
        """ Method for creating a single target point / orient constraint """

        if target is None:
            return

        targetDag = self.GetDagFromNameOrObject(target)
        slaveDag = self.GetDagFromNameOrObject(slave)

        sfm.PushSelection()
        self.SelectDagList([targetDag, slaveDag])

        pointConstraintTarget = sfm.PointConstraint(controls=bCreateControls)
        orientConstraintTarget = sfm.OrientConstraint(controls=bCreateControls)

        if group is not None:
            if pointConstraintTarget is not None:
                pointWeightControl = pointConstraintTarget.FindWeightControl()
                if pointWeightControl is not None:
                    group.AddControl(pointWeightControl)

            if orientConstraintTarget is not None:
                orientWeightControl = orientConstraintTarget.FindWeightControl()
                if orientWeightControl is not None:
                    group.AddControl(orientWeightControl)

        sfm.PopSelection()
        return

    def BuildArmLeg(self, rigPVTarget, rigEndTarget, bipStart, bipEnd, constrainEnd, group=None):
        """ Method for constraining an arm or leg to a set of handles """
        pvTargetDag = self.GetDagFromNameOrObject(rigPVTarget)
        endTargetDag = self.GetDagFromNameOrObject(rigEndTarget)
        startBoneDag = self.GetDagFromNameOrObject(bipStart)
        endBoneDag = self.GetDagFromNameOrObject(bipEnd)

        sfm.PushSelection()

        # Create a 2 bone ik constraint that constrains the two bones connecting the start and end
        sfm.SelectDag(endTargetDag)
        sfm.SelectDag(startBoneDag)
        sfm.SelectDag(endBoneDag)

        constraintTarget = sfm.IKConstraint(pvTarget=pvTargetDag, mo=True)

        # Add the control to the group if specified
        if (group is not None) and (constraintTarget is not None):
            control = constraintTarget.FindWeightControl()
            if control is not None:
                group.AddControl(control)

        # Orient constrain the foot to the rig handle so that it's rotation will not be effected by the ik
        if constrainEnd:
            sfm.ClearSelection()
            sfm.SelectDag(endBoneDag)
            sfm.SelectDag(endTargetDag)
            constraintTarget = sfm.OrientConstraint(mo=True)

            if (group is not None) and (constraintTarget is not None):
                control = constraintTarget.FindWeightControl()
                if control is not None:
                    group.AddControl(control)

        sfm.PopSelection()

        return

    @staticmethod
    def AddAttributeToElement(element, attrName, attrType, attrValue):
        """ Add an attribute of the specified type with the sepcified value to the element and set
        the value of the attribute to the provided value. If an attribute of the specified type and
        name already exist the value will be updated, if there is an attribute with the specified
        name but the type does not match the function will do nothing and return None, otherwise it
        will return the attribute which was created or updated. It is possible to just call
        element.SetValue( attrName, value ), but this may interpret the type of the value wrong."""

        attr = element.AddAttribute(attrName, attrType)
        if attr is not None:
            attr.SetValue(attrValue)

        return attr

    @staticmethod
    def Match(nodeName, targetDag):
        """ Match the position and orientation of the specified node with the target node """
        sfm.Move(0, 0, 0, nodeName, space="RefObject", refObject=targetDag)
        sfm.Rotate(0, 0, 0, nodeName, space="RefObject", refObject=targetDag)
        return

    @staticmethod
    def CreateHandleRelativeTo(handleName, targetName, x, y, z, bPosControl=True, bRotControl=True):
        """ Create a rig handle a position relative to the specified node """
        handlePos = sfm.GetPosition(targetName)
        handleRot = sfm.GetRotation(targetName)
        handlePos[0] = handlePos[0] + x
        handlePos[1] = handlePos[1] + y
        handlePos[2] = handlePos[2] + z
        handle = sfm.CreateRigHandle(handleName, pos=handlePos, rot=handleRot, posControl=bPosControl,
                                     rotControl=bRotControl)
        return handle

    @staticmethod
    def CreateWeightedConstraint(targetA, targetB, slave, weight):
        """ Method for creating a weighted two target point / orient constraint """
        weightA = 1 - weight
        weightB = weight
        sfm.PointConstraint(targetA, slave, w=weightA, mo=False)
        sfm.OrientConstraint(targetA, slave, w=weightA, mo=True)
        sfm.PointConstraint(targetB, slave, w=weightB, mo=False)
        sfm.OrientConstraint(targetB, slave, w=weightB, mo=True)
        return

    @staticmethod
    def CreateAttachmentHandleInGroup(attachmentName, controlGroup):
        """ Method for creating an attachment dag node and adding to the specified control and display groups """
        attachmentDag = sfm.CreateAttachmentHandle(attachmentName)
        if attachmentDag is not None:
            attachmentCtrl = attachmentDag.FindTransformControl()
            controlGroup.AddControl(attachmentCtrl)

        return attachmentDag

    @staticmethod
    def FindChildControlGroup(controlGroup, childName, bRecursive=False):
        """ Find the child control group of the specified control group, optionally search recursively
        to find the child of the specified name anywhere in the sub-tree of the specified control group."""
        if not controlGroup.HasChildGroup(childName, bRecursive):
            return None

        return controlGroup.FindChildByName(childName, bRecursive)

    @staticmethod
    def MoveControlGroup(groupName, oldParent, newParent):
        """ Method for moving a control group from one parent to another """

        # if ( oldParent.HasChildGroup( groupName, False ) == False ):
        #    return False;

        group = oldParent.FindChildByName(groupName, False)

        if group is not None:
            newParent.AddChild(group)
            return

        control = oldParent.FindControlByName(groupName, False)
        if control is not None:
            newParent.AddControl(control)

        return

    @staticmethod
    def AddDagControlsToGroup(group, *dagList):
        """ Add the transform controls of the specified dags to the group """
        if group is not None:
            for dag in dagList:
                control = dag.FindTransformControl()
                if control is not None:
                    group.AddControl(control)

        return

    @staticmethod
    def SetControlGroupColor(group, color):
        """ Method for setting the colors of a control group """
        if group is None:
            return

        group.SetGroupColor(color, True)
        colorRed = max((color.r() - 40), 0)
        colorBlue = max((color.g() - 40), 0)
        colorGreen = max((color.b() - 40), 0)
        colorDimmed = vs.Color(colorRed, colorBlue, colorGreen, 255)
        group.SetControlColor(colorDimmed, True)

    def SetControlGroupColorByName(self, groupName, rootGroup, color):
        """ Find the control group with the specified name and set its color """
        group = self.FindChildControlGroup(rootGroup, groupName, True)
        self.SetControlGroupColor(group, color)

    @staticmethod
    def GetChannelsClipForAnimSet(animSet, shot):
        """ Get the channel track group for the sepecified animation set """
        trackGroup = shot.FindTrackGroup("channelTrackGroup")
        track = trackGroup.FindOrAddTrack("animSetEditorChannels", vs.DMECLIP_CHANNEL)
        channelsClip = track.FindNamedClip(animSet.GetName())
        return channelsClip

    # Create a connection operator, set its input source and add it to the specified animation set

    def CreateConnection(self, name, srcElement, srcAttrName, animSet):
        newConnection = vs.CreateElement("DmeConnectionOperator", name, animSet.GetFileId())
        newConnection.SetInput(srcElement, srcAttrName)
        animSet.AddOperator(newConnection)
        self.AddElementToRig(newConnection, animSet)
        return newConnection

    # Create an expression operator and initalize it with the provided expression

    def CreateExpression(self, name, expression, animSet):
        newExpression = vs.CreateElement("DmeExpressionOperator", name, animSet.GetFileId())
        newExpression.expr = expression
        animSet.AddOperator(newExpression)
        self.AddElementToRig(newExpression, animSet)
        return newExpression

    # Create a rotation constraint and initalize it with the provided axis and rotation

    def CreateRotationConstraint(self, name, axis, slaveDag, animSet):
        rotationConstraint = vs.CreateElement("DmeRigRotationConstraintOperator", name, animSet.GetFileId())
        rotationConstraint.SetSlave(slaveDag)
        rotationConstraint.AddAxis(axis)
        rotationConstraint.DisconnectTransformChannels()
        animSet.AddOperator(rotationConstraint)
        self.AddElementToRig(rotationConstraint, animSet)
        return rotationConstraint

    # Create an element with an attribute of the spefified type that is driven by a control, returns a
    # tuple with the control and the value element

    def CreateControlledValue(self, controlName, attrName, type, value, animSet, shot):
        # Create the control and attach a value channel to it
        control = self.CreateControlAndChannel(controlName, type, value, animSet, shot)

        # Create an element to hold output value of the channel
        valueElementName = controlName + "_value"
        valueElement = vs.CreateElement("DmElement", valueElementName, animSet.GetFileId());
        valueElement.SetValue(attrName, 1.0)
        control.channel.SetOutput(valueElement, attrName)

        self.AddElementToRig(valueElement, animSet)

        return control, valueElement

    # Add the children of the specified node to the current selection

    @staticmethod
    def SelectChildren(dagNode):
        numChildren = dagNode.children.Count()
        for i in range(numChildren):
            child = dagNode.children[i]
            sfm.SelectDag(child)

        return

    def AddControlsInGroupToDisplaySet(self, controlGroup, displaySet):
        """ Add all of the controls in the specified group and its childern to the specified display set,
        this will result in removing those controls from any display set they are currently apart of."""

        if controlGroup is not None:
            for childGroup in controlGroup.children:
                self.AddControlsInGroupToDisplaySet(childGroup, displaySet)

        for control in controlGroup.controls:
            sfm.AddControlToDisplaySet(control, displaySet)

        return

    def AddControlGroupsToDisplaySet(self, controlGroupList, displaySet):
        """ Find the control groups matching each name in the provided list and add all of the control
        in each group to the specified display set"""

        animSet = sfm.GetCurrentAnimationSet()
        rootGroup = animSet.GetRootControlGroup()

        for groupName in controlGroupList:
            if rootGroup.HasChildGroup(groupName, True):
                controlGroup = rootGroup.FindChildByName(groupName, True)
                self.AddControlsInGroupToDisplaySet(controlGroup, displaySet)
            else:
                control = rootGroup.FindControlByName(groupName, True)
                if control is not None:
                    sfm.AddControlToDisplaySet(control, displaySet)

        return

    def CreateDisplaySetWithControlGroups(self, displaySetName, controlGroupList, bHidden=False, bSnap=False):
        """ Create a display set and add the controls contained in each of the groups specified by name
        in the provided list. Set the hidden and snap options of the display set"""

        newDisplaySet = sfm.CreateDisplaySet(displaySetName)
        newDisplaySet.SetHidden(bHidden)
        newDisplaySet.SetSnap(bSnap)

        self.AddControlGroupsToDisplaySet(controlGroupList, newDisplaySet)

        return newDisplaySet

    @staticmethod
    def CreateModelAnimationSet(baseName, modelPath):
        """ Create a model and animation set for that model and add the model to the scene"""

        shot = sfm.GetCurrentShot()
        model = sfm.CreateModel(modelPath)
        if model is not None:
            animSet = sfm.CreateAnimationSet(baseName, target=model)
            if animSet is not None:
                dag = vs.CreateElement("DmeDag", baseName, shot.GetFileId())
                dag.AddChild(model)
                shot.scene.AddChild(dag)

        return animSet

    @staticmethod
    def SelectDagList(dagList):
        """ Add each of the dags in the specified list to the current selection """
        for dagNode in dagList[::-1]:
            if dagNode is not None:
                sfm.SelectDag(dagNode)

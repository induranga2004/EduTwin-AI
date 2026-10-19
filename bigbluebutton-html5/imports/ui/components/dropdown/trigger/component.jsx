import React, { Component } from 'react';
import Button from '/imports/ui/components/common/button/component';
import PropTypes from 'prop-types';
import cx from 'classnames';

import KEYS from '/imports/utils/keys';

const propTypes = {
  children: PropTypes.element.isRequired,
};

export default class DropdownTrigger extends Component {
  static isButtonTriggerOnEmoji(buttonComponent) {
    return (
      buttonComponent
      && (buttonComponent.type === Button)
      && (buttonComponent.props)
      && (buttonComponent.props.children)
    );
  }

  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.trigger = null;
  }

  handleClick(e) {
    e.stopPropagation();
    const { dropdownToggle, onClick } = this.props;
    if (onClick) onClick();
    return dropdownToggle();
  }

  handleKeyDown(event) {
    event.stopPropagation();
    const { dropdownShow, dropdownHide } = this.props;
    const { key } = event;

    if ([KEYS.SPACE, KEYS.ENTER].includes(key)) {
      event.preventDefault();
      event.stopPropagation();
    } else if ([KEYS.ARROW_UP, KEYS.ARROW_DOWN].includes(key)) {
      dropdownShow();
    } else if (KEYS.ESCAPE === key) {
      dropdownHide();
    }
  }

  render() {
    const { dropdownIsOpen } = this.props;
    const remainingProps = { ...this.props };
    delete remainingProps.dropdownToggle;
    delete remainingProps.dropdownShow;
    delete remainingProps.dropdownHide;
    delete remainingProps.dropdownIsOpen;

    const {
      children,
      className,
      ...restProps
    } = remainingProps;

    let TriggerComponent;
    let TriggerComponentBounded;

    const buttonComponentProps = {
      ...restProps,
      'aria-expanded': dropdownIsOpen,
    };

    const triggerComponentProps = {
      onClick: this.handleClick,
      onKeyDown: this.handleKeyDown,
    };

    if (DropdownTrigger.isButtonTriggerOnEmoji(children)) {
      const { children: grandChildren } = children.props;

      triggerComponentProps.className = cx(children.props.className, className);

      TriggerComponent = React.Children.only(grandChildren);
      TriggerComponentBounded = React.cloneElement(TriggerComponent,
        triggerComponentProps);

      const ButtonComponent = React.Children.only(children);
      return React.cloneElement(ButtonComponent,
        buttonComponentProps, TriggerComponentBounded);
    }

    buttonComponentProps.className = cx(children.props.className, className);
    TriggerComponent = React.Children.only(children);

    TriggerComponentBounded = React.cloneElement(TriggerComponent,
      { ...buttonComponentProps, ...triggerComponentProps });

    return TriggerComponentBounded;
  }
}

DropdownTrigger.propTypes = propTypes;
